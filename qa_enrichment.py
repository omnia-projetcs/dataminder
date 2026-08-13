"""Post-dedup Q&A enrichment for Dataminder."""

from __future__ import annotations

import json
import os
import random
import time

from tqdm import tqdm

from json_repair import _try_parse_enrichment_json
from llm_client import LLMClient
from processing_manifest import atomic_write_text


_ENRICHMENT_PROMPTS = {
    "cyber": (
        "You are a cybersecurity expert. Transform the provided Q&A into a structured report. "
        "Return ONLY a valid JSON object with exactly two keys: \"input\" and \"output\". "
        "\"input\" = credible technical context (logs, tool output, scenario). "
        "\"output\" = structured report with: summary, MITRE ATT&CK (TXXXX), CVSS, recommendations, IOC. "
        "No markdown, no explanation, no code fences — just the raw JSON object."
    ),
    "finance": (
        "You are a finance and markets expert. Transform the provided Q&A into a structured report. "
        "Return ONLY a valid JSON object with exactly two keys: \"input\" and \"output\". "
        "\"input\" = credible financial context (market data, filings, or a scenario). "
        "\"output\" = structured report with: summary, instruments or metrics, risk factors, "
        "regulatory notes only when implied by the Q&A, and recommendations. "
        "Do not invent tickers, prices, or regulations. "
        "No markdown, no explanation, no code fences — just the raw JSON object."
    ),
    "generic": (
        "You are a technical expert. Transform the provided Q&A into a structured report. "
        "Return ONLY a valid JSON object with exactly two keys: \"input\" and \"output\". "
        "\"input\" = realistic technical context. "
        "\"output\" = structured report with: summary, key facts, constraints, and recommendations. "
        "No markdown, no explanation, no code fences — just the raw JSON object."
    ),
}


def _enrich_after_dedup(qa_list, ratio=0.3, model_name="gemma3:4b-it-q4_K_M", llm_client=None, output_path=None, domain="cyber"):
    """Enrich a fraction of the deduplicated dataset just before saving.

    Uses the shared LLMClient abstraction so enrichment works with both
    Ollama and vLLM providers.

    Resume support: if output_path exists, loads it and skips entries
    already enriched. Saves incrementally (periodically and atomically) 
    to support resume on crash.
    """
    if ratio <= 0 or not qa_list:
        return qa_list

    if llm_client is None:
        llm_client = LLMClient(provider="ollama")
    if domain not in _ENRICHMENT_PROMPTS:
        raise ValueError("domain must be one of: cyber, finance, generic")

    # Resume: load existing enriched data if available
    already_enriched_count = 0
    if output_path and os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if isinstance(existing, list) and len(existing) == len(qa_list):
                # Restore already-enriched entries
                for i, entry in enumerate(existing):
                    if isinstance(entry, dict) and isinstance(entry.get("_meta"), dict) and entry["_meta"].get("enriched"):
                        qa_list[i] = entry
                        already_enriched_count += 1
                if already_enriched_count > 0:
                    print(f"  Resuming enrichment: {already_enriched_count} entries already enriched.")
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [WARNING] Could not load existing enriched file for resume: {e}")

    # Determine targets and indices
    n_to_enrich = max(1, int(len(qa_list) * ratio))
    target_new_to_enrich = n_to_enrich - already_enriched_count

    if target_new_to_enrich <= 0:
        print(f"\nPhase 5: All {already_enriched_count} target entries already enriched (target ratio {ratio*100:.0f}% met). Skipping.")
        return qa_list

    # Find all indices that are NOT yet enriched
    non_enriched_indices = [
        i for i, entry in enumerate(qa_list)
        if not (isinstance(entry, dict) and isinstance(entry.get("_meta"), dict) and entry["_meta"].get("enriched"))
    ]

    # Sample exactly what is needed to reach the target ratio
    rng = random.Random(getattr(llm_client, "seed", None))
    indices = rng.sample(non_enriched_indices, min(target_new_to_enrich, len(non_enriched_indices)))

    if not indices:
        print("\nPhase 5: No more entries available to enrich. Skipping.")
        return qa_list

    system_prompt = _ENRICHMENT_PROMPTS[domain]

    total_target = already_enriched_count + len(indices)
    print(f"\nPhase 5: Post-deduplication enrichment: {len(indices)} remaining / {total_target} target ({ratio*100:.0f}%)...")

    enriched_count = 0
    
    def _save_progress():
        if output_path:
            try:
                atomic_write_text(
                    output_path,
                    json.dumps(qa_list, ensure_ascii=False, indent=2) + "\n",
                )
            except Exception as save_err:
                tqdm.write(f"  [WARNING] Failed to save progress: {save_err}")

    try:
        for idx in tqdm(indices, desc="Enriching", unit="entry"):
            entry = qa_list[idx]
            if not isinstance(entry, dict):
                continue
            inst = entry.get("instruction", entry.get("question", ""))
            ans = entry.get("output", entry.get("answer", ""))
            if not inst or not ans:
                continue

            try:
                content = llm_client.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Question: {inst}\nAnswer: {ans}\n\nGenerate enriched input/output."}
                    ],
                    keep_alive=-1
                )
                result = _try_parse_enrichment_json(content)
                if result and "input" in result and "output" in result:
                    existing_meta = qa_list[idx].get("_meta", {})
                    if not isinstance(existing_meta, dict):
                        existing_meta = {}
                    qa_list[idx] = {
                        "instruction": inst,
                        "input": result["input"],
                        "output": result["output"] if isinstance(result["output"], str) else json.dumps(result["output"], ensure_ascii=False),
                        "_meta": {
                            **existing_meta,
                            "enriched": True,
                            "model": model_name,
                            "domain": domain,
                        }
                    }
                    enriched_count += 1
                    
                    # Save periodically (every 50 successful enrichments) to avoid excessive disk/CPU thrashing
                    if enriched_count % 50 == 0:
                        _save_progress()
                else:
                    preview = content[:300].replace('\n', ' ') if content else '<empty>'
                    tqdm.write(f"  [SKIP] [{idx}] Could not parse enrichment JSON from LLM response. Preview: {preview}")
            except Exception as e:
                tqdm.write(f"  [FAIL] [{idx}] Error: {e}")
            time.sleep(0.2)
    finally:
        # Final save on exit/interruption to ensure no progress is lost
        if enriched_count > 0:
            _save_progress()

    final_total = sum(1 for e in qa_list if isinstance(e, dict) and isinstance(e.get("_meta"), dict) and e["_meta"].get("enriched"))
    print(f"  Enrichment complete. {enriched_count} new + {already_enriched_count} resumed = {final_total} total enriched.\n")
    return qa_list
