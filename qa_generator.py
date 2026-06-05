import os
import json
import re
import signal
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import difflib
import time
import random
from llm_client import LLMClient
from logger import log_error as _log_error

CHUNK_SIZE = 5000

# --- Graceful shutdown on SIGTERM / SIGINT ---
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT by setting a flag so the processing loop
    can finish the current chunk, flush to disk, and exit cleanly."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    if not _shutdown_requested:
        _shutdown_requested = True
        tqdm.write(f"\n  [SIGNAL] Received {sig_name} — finishing current chunk and saving progress...")
    else:
        tqdm.write(f"\n  [SIGNAL] Received {sig_name} again — forcing exit.")
        sys.exit(1)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def _split_into_chunks(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks at paragraph or line boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        if len(text) - start <= chunk_size:
            chunks.append(text[start:])
            break

        # Try to find a paragraph break to split cleanly
        split_point = text.rfind('\n\n', start, start + chunk_size)
        if split_point == -1 or split_point <= start:
            # Fallback to newline
            split_point = text.rfind('\n', start, start + chunk_size)
        if split_point == -1 or split_point <= start:
            # Fallback to hard split
            split_point = start + chunk_size

        chunks.append(text[start:split_point].strip())
        # Advance past the split delimiter to guarantee forward progress
        start = split_point + 1
    return chunks

def log_error(filepath, error_msg):
    _log_error(filepath, error_msg, category="QA GENERATION")

def _sanitize_json_string(raw_json):
    """Sanitize a JSON string that may contain literal control characters
    (newlines, tabs, etc.) inside string values, which is invalid JSON.
    This replaces unescaped control characters with their escape sequences."""
    # Replace literal control characters that break JSON parsing
    # We need to be careful: \n that is already escaped (\\n in the raw string) should stay.
    # Strategy: process character by character tracking if we're inside a JSON string value.
    
    result = []
    in_string = False
    i = 0
    while i < len(raw_json):
        ch = raw_json[i]
        
        if ch == '\\' and in_string and i + 1 < len(raw_json):
            # Escaped character inside a string — keep both characters as-is
            result.append(ch)
            result.append(raw_json[i + 1])
            i += 2
            continue
        
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue
        
        if in_string:
            # Replace literal control characters with their JSON escape sequences
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
        else:
            result.append(ch)
        
        i += 1
    
    return ''.join(result)


def _try_parse_json(content):
    """Try multiple strategies to extract a JSON array from LLM output."""
    
    # Strategy 1: Extract JSON from markdown code block ```json ... ```
    code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
        sanitized = _sanitize_json_string(json_str)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Find the outermost [...] array
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if match:
        json_str = match.group(0)
        sanitized = _sanitize_json_string(json_str)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Try direct parse of the full content
    sanitized = _sanitize_json_string(content)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Line-by-line recovery — extract individual {"question":..., "answer":...} objects
    pairs = []
    for obj_match in re.finditer(r'\{[^{}]*?"question"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"answer"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', content, re.DOTALL):
        try:
            q = obj_match.group(1).replace('\\n', ' ').replace('\\t', ' ').strip()
            a = obj_match.group(2).replace('\\n', ' ').replace('\\t', ' ').strip()
            if q and a:
                pairs.append({"question": q, "answer": a})
        except Exception:
            continue
    
    return pairs if pairs else None


def generate_qa_from_text(text, model_name="gemma3:12b", source_file="N/A", llm_client=None):
    if llm_client is None:
        llm_client = LLMClient(provider="ollama")

    prompt = f"""
You are an expert AI dataset creator specializing in technical cybersecurity and IT training data. Based on the following document, generate a list of high-quality Question/Answer pairs for fine-tuning an AI model. Prioritize technical, hands-on questions but also include conceptual questions when the content warrants it.

CRITICAL RULES:
1. PRIORITIZE highly technical and specific questions that test real-world, hands-on knowledge. Include concrete details such as tool names, command syntax, protocol specifics, CVE references, configuration parameters, registry keys, API calls, or attack technique names when available.
2. Conceptual questions are acceptable but should remain specific and non-trivial. Avoid overly generic questions like "What is cybersecurity?" or "Why is security important?".
3. Answers must be self-contained, precise, and technically accurate. Include specific values, commands, paths, or configurations when relevant.
4. Keep answers in plain text. Do NOT include code blocks, markdown formatting, or newlines inside answers.
5. NEVER reference the source document, book, author, chapter, or publication. Write questions and answers as standalone technical knowledge.
6. Be EXHAUSTIVE. Extract as many relevant, high-quality Question/Answer pairs as possible from the provided text. Ensure no important technical details, configurations, or concepts are omitted.
7. Provide the output STRICTLY as a JSON array of objects. No other text.
Format:
[
  {{"question": "What is X?", "answer": "X is Y because Z."}},
  {{"question": "How does A work?", "answer": "A works by doing B."}}
]

Document text:
---
{text}
---
"""


    try:
        content = llm_client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            keep_alive=-1
        )
        
        parsed = _try_parse_json(content)
        
        if parsed is None:
            tqdm.write(f"  [WARNING] Could not parse JSON from AI response. Raw preview: {content[:200]}")
            log_error(source_file, f"JSON parse failed. Raw response: {content[:500]}")
            return []
        
        if not isinstance(parsed, list):
            tqdm.write(f"  [WARNING] AI returned non-list type: {type(parsed)}")
            return []
            
        return parsed
            
    except Exception as e:
        tqdm.write(f"  [ERROR] LLM call failed: {e}")
        log_error(source_file, f"LLM exception: {e}")
        return []

# Regex patterns that detect references to source books/documents/authors
_SOURCE_REF_PATTERNS = re.compile(
    r'(?:'
    r'(?:the|this|that|a)\s+(?:book|document|text|article|paper|publication|manual|guide|handbook|report|chapter|module|section|course|training|material|slide|presentation|lecture|reference)'
    r'|(?:according\s+to|as\s+(?:described|explained|stated|mentioned|discussed|noted|outlined|covered|presented|defined|highlighted)\s+(?:in|by))'
    r'|(?:the\s+author(?:s)?\s+(?:describe|explain|state|mention|discuss|note|outline|present|define|highlight|argue|suggest|recommend|emphasize|propose))'
    r'|\bthe\s+book\s+\*'
    r'|\*[A-Z][^*]{3,60}\*'  # catches *Book Title* in italics
    r'|(?:in\s+chapter\s+\d)'
    r'|(?:in\s+module\s+\d)'
    r'|(?:in\s+section\s+\d)'
    r'|(?:target\s+audience(?:s)?\s+(?:for|of)\s+the)'
    r'|(?:what\s+(?:does|do|did)\s+the\s+(?:book|document|text|author|manual|guide|chapter|module))'
    r'|(?:how\s+does\s+the\s+(?:book|document|text|author|manual|guide|chapter|module))'
    r')',
    re.IGNORECASE
)

def _references_source_material(qa):
    """Return True if the Q&A pair references a source book/document/author."""
    text = qa.get('question', '') + ' ' + qa.get('answer', '')
    return bool(_SOURCE_REF_PATTERNS.search(text))

def _get_ngrams(text, n=3):
    """Generate character n-grams (shingles) from text."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _minhash_signature(shingles, num_hashes=64):
    """Compute a MinHash signature for a set of shingles.
    Uses Python's built-in hash() with seed mixing instead of hashlib.md5
    for ~20-50x faster hashing (non-cryptographic, fine for similarity)."""
    signature = []
    for i in range(num_hashes):
        min_hash = float('inf')
        for shingle in shingles:
            h = hash((i, shingle)) & 0xFFFFFFFFFFFFFFFF  # ensure positive
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return tuple(signature)


def _lsh_buckets(signature, num_bands=16):
    """Split a MinHash signature into bands for LSH bucketing."""
    band_size = len(signature) // num_bands
    bands = []
    for i in range(num_bands):
        band = signature[i * band_size:(i + 1) * band_size]
        bands.append(hash(band))
    return bands


def deduplicate_qa(qa_list, threshold=0.85):
    if not qa_list:
        return []

    num_hashes = 64
    num_bands = 16

    # Phase 1: Exact dedup via normalized text hash
    seen_exact = {}
    deduped_exact = []
    for qa in qa_list:
        key = qa.get("question", "").lower().strip()
        if key not in seen_exact:
            seen_exact[key] = True
            deduped_exact.append(qa)

    exact_removed = len(qa_list) - len(deduped_exact)
    if exact_removed:
        tqdm.write(f"  Removed {exact_removed} exact duplicates.")

    # Phase 2: Build MinHash signatures + LSH index (cached)
    print(f"  Building similarity index for {len(deduped_exact)} pairs...")
    signatures = []
    all_bands = []  # pre-cache LSH bands to avoid recomputing in phase 3
    for qa in tqdm(deduped_exact, desc="Indexing", unit="pair"):
        shingles = _get_ngrams(qa.get("question", ""), n=3)
        sig = _minhash_signature(shingles, num_hashes)
        signatures.append(sig)
        all_bands.append(_lsh_buckets(sig, num_bands))

    # Build LSH band buckets: (band_idx, band_hash) -> list of indices
    band_buckets = defaultdict(list)
    for idx, bands in enumerate(all_bands):
        for band_idx, band_hash in enumerate(bands):
            band_buckets[(band_idx, band_hash)].append(idx)

    # Phase 3: For each pair, check only LSH candidates with SequenceMatcher
    removed = set()
    for idx in tqdm(range(len(deduped_exact)), desc="Deduplicating", unit="pair"):
        if idx in removed:
            continue

        # Collect candidate indices from shared LSH bands (using cached bands)
        candidates = set()
        for band_idx, band_hash in enumerate(all_bands[idx]):
            for cand_idx in band_buckets[(band_idx, band_hash)]:
                if cand_idx > idx and cand_idx not in removed:
                    candidates.add(cand_idx)

        if not candidates:
            continue

        q_text = deduped_exact[idx].get("question", "").lower()
        for cand_idx in candidates:
            if cand_idx in removed:
                continue
            cand_text = deduped_exact[cand_idx].get("question", "").lower()
            if difflib.SequenceMatcher(None, q_text, cand_text).ratio() > threshold:
                removed.add(cand_idx)

    unique_qa = [qa for idx, qa in enumerate(deduped_exact) if idx not in removed]
    return unique_qa

def _load_already_processed(raw_path):
    """Read the raw JSONL file and return chunk-level resume state.

    Returns:
        dict[str, set[int]]: mapping of source filename -> set of chunk indices
        already processed.  Entries without a ``_chunk_idx`` field (legacy format)
        are treated as chunk -1 which causes the whole file to be considered done.
    """
    done = defaultdict(set)
    if not os.path.exists(raw_path):
        return done
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                src = obj.get("_source_file", "")
                if src:
                    chunk_idx = obj.get("_chunk_idx", -1)
                    done[src].add(chunk_idx)
            except json.JSONDecodeError:
                continue
    return done


def _load_raw_pairs(raw_path):
    """Read all Q&A pairs from the raw JSONL file."""
    pairs = []
    if not os.path.exists(raw_path):
        return pairs
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "question" in obj and "answer" in obj:
                    pairs.append({"question": obj["question"], "answer": obj["answer"]})
            except json.JSONDecodeError:
                continue
    return pairs


def generate_qa_dataset(source_dir, dest_dir, model_name, llm_client=None, num_threads=1, force=False, enrich=False, enrich_ratio=0.3):
    if llm_client is None:
        llm_client = LLMClient(provider="ollama")

    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist. Creating it now...")
        os.makedirs(source_dir, exist_ok=True)

    os.makedirs(dest_dir, exist_ok=True)
    
    raw_jsonl_path = os.path.join(dest_dir, "dataset_qa_raw.jsonl")
    
    if force and os.path.exists(raw_jsonl_path):
        try:
            os.remove(raw_jsonl_path)
            print(f"Force option enabled: resetting raw Q&A data at {raw_jsonl_path}")
        except Exception as e:
            print(f"Warning: Could not remove {raw_jsonl_path}: {e}")
    
    files_to_process = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.md'):
                files_to_process.append(os.path.join(root, file))
                
    if not files_to_process:
        print(f"No Markdown (.md) files found in '{source_dir}'.")
        return
    
    # Check which files/chunks were already processed (resume support)
    already_done = _load_already_processed(raw_jsonl_path) if not force else defaultdict(set)
    # Files where legacy entries (no _chunk_idx) exist are considered fully done
    fully_done_files = {f for f, chunks in already_done.items() if -1 in chunks}
    partially_done_files = {f for f in already_done if f not in fully_done_files}
    if already_done:
        total_before = len(files_to_process)
        # Remove fully-processed files; keep partially-processed ones for chunk-level resume
        files_to_process = [f for f in files_to_process if os.path.basename(f) not in fully_done_files]
        skipped = total_before - len(files_to_process)
        partial_msg = f" ({len(partially_done_files)} partially done — will resume at chunk level)" if partially_done_files else ""
        print(f"Resuming: {skipped} files already processed, {len(files_to_process)} remaining.{partial_msg}")
        
    dataset_json_path = os.path.join(dest_dir, "dataset_qa.json")
    dataset_md_path = os.path.join(dest_dir, "dataset_qa.md")
    
    if not force and len(files_to_process) == 0:
        if os.path.exists(dataset_json_path) and os.path.exists(dataset_md_path):
            print("Resuming: Q&A dataset is already complete and all files have been processed. Skipping completely.")
            llm_client.unload_model(model_name)
            return
    
    # Count existing pairs without loading them all into memory
    total_saved = 0
    if os.path.exists(raw_jsonl_path):
        with open(raw_jsonl_path, 'r', encoding='utf-8') as f:
            total_saved = sum(1 for line in f if line.strip())
        
    if files_to_process:
        print(f"Processing {len(files_to_process)} Markdown files for Q&A generation...")
        if num_threads > 1:
            print(f"Using {num_threads} threads for parallel chunk processing.")
        
        failed_files = 0
        pbar = tqdm(files_to_process, desc="Generating Q&A", unit="file")
             
        for input_path in pbar:
            filename = os.path.basename(input_path)
            pbar.set_postfix({"file": filename[:20], "q_saved": total_saved})
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                if not text.strip():
                    log_error(input_path, "File is empty.")
                    continue
                    
                # Split text into chunks to prevent context blowup while processing entire files
                chunks = _split_into_chunks(text)
                
                # Determine which chunks were already processed (chunk-level resume)
                done_chunks = already_done.get(filename, set())
                
                file_count = 0
                filtered_count = 0
                chunk_errors = 0

                def _flush_chunk_pairs(chunk_pairs, chunk_idx):
                    """Write Q&A pairs from one chunk to disk immediately."""
                    nonlocal file_count, filtered_count, total_saved
                    if not isinstance(chunk_pairs, list):
                        return
                    with open(raw_jsonl_path, 'a', encoding='utf-8') as f_raw:
                        for qa in chunk_pairs:
                            if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                                if _references_source_material(qa):
                                    filtered_count += 1
                                    continue
                                record = {
                                    "question": qa['question'],
                                    "answer": qa['answer'],
                                    "_source_file": filename,
                                    "_chunk_idx": chunk_idx
                                }
                                f_raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                                file_count += 1
                                total_saved += 1

                if num_threads > 1 and len(chunks) > 1:
                    # Parallel chunk processing — flush per chunk as results arrive
                    futures = {}
                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                        for chunk_idx, chunk in enumerate(chunks):
                            if not chunk or chunk_idx in done_chunks:
                                continue
                            future = executor.submit(
                                generate_qa_from_text, chunk,
                                model_name=model_name,
                                source_file=input_path,
                                llm_client=llm_client
                            )
                            futures[future] = chunk_idx

                        completed = 0
                        for future in as_completed(futures):
                            chunk_idx = futures[future]
                            completed += 1
                            pbar.set_postfix({"file": filename[:20], "chunk": f"{completed}/{len(futures)}", "q_saved": total_saved})
                            try:
                                chunk_pairs = future.result()
                                _flush_chunk_pairs(chunk_pairs, chunk_idx)
                            except Exception as e:
                                chunk_errors += 1
                                log_error(input_path, f"Chunk {chunk_idx}: {e}")
                                tqdm.write(f"  [{filename}] chunk {chunk_idx} error: {e}")
                else:
                    # Sequential chunk processing (default) — flush after each chunk
                    for chunk_idx, chunk in enumerate(chunks):
                        if _shutdown_requested:
                            tqdm.write(f"\n  [SHUTDOWN] Stopping gracefully. Progress saved ({total_saved} pairs on disk). Resume will pick up here.")
                            pbar.close()
                            print(f"\nGraceful shutdown complete. {total_saved} total Q&A pairs saved to: {raw_jsonl_path}")
                            print("Re-run the same command to resume from where you left off.")
                            llm_client.unload_model(model_name)
                            return
                        if not chunk:
                            continue
                        if chunk_idx in done_chunks:
                            continue
                        if len(chunks) > 1:
                            pbar.set_postfix({"file": filename[:20], "chunk": f"{chunk_idx+1}/{len(chunks)}", "q_saved": total_saved})
                        
                        try:
                            chunk_pairs = generate_qa_from_text(chunk, model_name=model_name, source_file=input_path, llm_client=llm_client)
                            _flush_chunk_pairs(chunk_pairs, chunk_idx)
                        except Exception as e:
                            chunk_errors += 1
                            log_error(input_path, f"Chunk {chunk_idx}: {e}")
                            tqdm.write(f"  [{filename}] chunk {chunk_idx} error: {e}")
                            continue

                if file_count == 0 and chunk_errors == 0:
                    failed_files += 1
                    tqdm.write(f"  [{filename}] No valid Q&A pairs extracted.")
                elif file_count > 0:
                    msg = f"  [{filename}] +{file_count} pairs saved"
                    if filtered_count:
                        msg += f" ({filtered_count} filtered: source refs)"
                    if chunk_errors:
                        msg += f" ({chunk_errors} chunk errors)"
                    msg += f" (total: {total_saved})"
                    tqdm.write(msg)
                else:
                    failed_files += 1
                    log_error(input_path, f"All {chunk_errors} chunks failed.")
            except Exception as e:
                failed_files += 1
                log_error(input_path, f"Unexpected exception: {str(e)}")
                tqdm.write(f"  [{filename}] Error: {str(e)}")
                continue

        print(f"\nPhase 1 Complete. {total_saved} total raw Q&A pairs on disk ({failed_files} files had issues).")
    else:
        print("All files already processed. Skipping to deduplication.")
    
    # Load all raw pairs from disk for deduplication
    all_qa_pairs = _load_raw_pairs(raw_jsonl_path)
    
    if not all_qa_pairs:
        print("ERROR: No Q&A pairs were generated at all! Check error.log for details.")
        print("Common causes: LLM server not running, model not available, or all AI responses were unparseable.")
        return
    
    # Phase 2: Filter out any remaining source references from raw data
    print(f"Phase 2: Filtering source references from {len(all_qa_pairs)} pairs...")
    clean_pairs = [qa for qa in all_qa_pairs if not _references_source_material(qa)]
    filtered_refs = len(all_qa_pairs) - len(clean_pairs)
    if filtered_refs:
        print(f"  Removed {filtered_refs} pairs referencing source material.")
    
    # Phase 3: Deduplication
    print(f"Phase 3: Removing duplicates from {len(clean_pairs)} pairs (similarity threshold 85%)...")
    unique_qa_pairs = deduplicate_qa(clean_pairs, threshold=0.85)
    
    dataset_json_path = os.path.join(dest_dir, "dataset_qa.json")
    dataset_md_path = os.path.join(dest_dir, "dataset_qa.md")
    
    print("Phase 4: Saving final clean files...")
    
    # Standard Alpaca json format (built before the with-block so it's available for enrichment)
    alpaca_format = [{"instruction": qa["question"], "input": "", "output": qa["answer"]} for qa in unique_qa_pairs]

    # Save JSON array
    with open(dataset_json_path, 'w', encoding='utf-8') as f_json:
        json.dump(alpaca_format, f_json, ensure_ascii=False, indent=2)
        
    # Save Markdown
    with open(dataset_md_path, 'w', encoding='utf-8') as f_md:
        f_md.write("# QA Dataset\n\n")
        for qa in unique_qa_pairs:
            f_md.write(f"**Q: {qa['question']}**\n\n")
            f_md.write(f"**A:** {qa['answer']}\n\n")
            f_md.write("---\n\n")

    print(f"\nQ&A Generation Complete! Generated {len(unique_qa_pairs)} unique Q&A pairs (Removed {len(all_qa_pairs) - len(unique_qa_pairs)} duplicates).")
    print(f"Saved JSON dataset to: {dataset_json_path}")
    print(f"Saved Markdown readable dataset to: {dataset_md_path}")
    print(f"Raw data preserved in: {raw_jsonl_path}")

    # Phase 5: Post-deduplication enrichment (only in full pipeline mode)
    if enrich and enrich_ratio > 0:
        dataset_enriched_path = os.path.join(dest_dir, "dataset_qa_enriched.json")
        enriched_pairs = _enrich_after_dedup(list(alpaca_format), ratio=enrich_ratio, model_name=model_name, llm_client=llm_client, output_path=dataset_enriched_path)
        # Final save (also done incrementally inside the function for crash safety)
        with open(dataset_enriched_path, 'w', encoding='utf-8') as f_enriched:
            json.dump(enriched_pairs, f_enriched, ensure_ascii=False, indent=2)
        enriched_count = sum(1 for e in enriched_pairs if isinstance(e.get("_meta"), dict) and e["_meta"].get("enriched"))
        print(f"Saved enriched dataset ({enriched_count} entries enriched) to: {dataset_enriched_path}")
    elif enrich:
        print("\nPhase 5: Enrichment skipped (enrich-ratio is 0).")

    # Unload model from VRAM (only for Ollama)
    llm_client.unload_model(model_name)


def _enrich_after_dedup(qa_list, ratio=0.3, model_name="gemma3:12b", llm_client=None, output_path=None):
    """Enrich a fraction of the deduplicated dataset just before saving.

    Uses the shared LLMClient abstraction so enrichment works with both
    Ollama and vLLM providers.

    Resume support: if output_path exists, loads it and skips entries
    already enriched. Saves incrementally after each successful enrichment.
    """
    if ratio <= 0 or not qa_list:
        return qa_list

    if llm_client is None:
        llm_client = LLMClient(provider="ollama")

    # Resume: load existing enriched data if available
    if output_path and os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if isinstance(existing, list) and len(existing) == len(qa_list):
                # Restore already-enriched entries
                already_done = 0
                for i, entry in enumerate(existing):
                    if isinstance(entry.get("_meta"), dict) and entry["_meta"].get("enriched"):
                        qa_list[i] = entry
                        already_done += 1
                if already_done > 0:
                    print(f"  Resuming enrichment: {already_done} entries already enriched.")
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [WARNING] Could not load existing enriched file for resume: {e}")

    n_to_enrich = max(1, int(len(qa_list) * ratio))
    indices = random.sample(range(len(qa_list)), min(n_to_enrich, len(qa_list)))

    # Filter out indices that are already enriched (resume support)
    indices = [i for i in indices if not (isinstance(qa_list[i].get("_meta"), dict) and qa_list[i]["_meta"].get("enriched"))]

    if not indices:
        enriched_total = sum(1 for e in qa_list if isinstance(e.get("_meta"), dict) and e["_meta"].get("enriched"))
        print(f"\nPhase 5: All {enriched_total} target entries already enriched. Skipping.")
        return qa_list

    system_prompt = (
        "You are a cybersecurity expert. Transform the provided Q&A into a structured report. "
        "Return ONLY a valid JSON object with exactly two keys: \"input\" and \"output\". "
        "\"input\" = credible technical context (logs, tool output, scenario). "
        "\"output\" = structured report with: summary, MITRE ATT&CK (TXXXX), CVSS, recommendations, IOC. "
        "No markdown, no explanation, no code fences — just the raw JSON object."
    )

    already_enriched = sum(1 for e in qa_list if isinstance(e.get("_meta"), dict) and e["_meta"].get("enriched"))
    total_target = already_enriched + len(indices)
    print(f"\nPhase 5: Post-deduplication enrichment: {len(indices)} remaining / {total_target} target ({ratio*100:.0f}%)...")

    enriched_count = 0
    for idx in tqdm(indices, desc="Enriching", unit="entry"):
        entry = qa_list[idx]
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
            start, end = content.find('{'), content.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
                if "input" in result and "output" in result:
                    qa_list[idx] = {
                        "instruction": inst,
                        "input": result["input"],
                        "output": result["output"],
                        "_meta": {"enriched": True, "model": model_name}
                    }
                    enriched_count += 1
                    # Save incrementally to support resume on crash
                    if output_path:
                        with open(output_path, 'w', encoding='utf-8') as f_save:
                            json.dump(qa_list, f_save, ensure_ascii=False, indent=2)
        except Exception as e:
            tqdm.write(f"  [FAIL] [{idx}] Error: {e}")
        time.sleep(0.2)

    final_total = sum(1 for e in qa_list if isinstance(e.get("_meta"), dict) and e["_meta"].get("enriched"))
    print(f"  Enrichment complete. {enriched_count} new + {already_enriched} resumed = {final_total} total enriched.\n")
    return qa_list