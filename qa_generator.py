import os
import json
import re
import hashlib
import signal
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import difflib
import time
import random
from llm_client import LLMClient
from logger import log_error as _log_error
from document_ir import sha256_file
from processing_manifest import atomic_write_text, relative_source_id, stable_fingerprint
from dataset_export import clean_dataset, prepare_hf_dataset


CHUNK_SIZE = 5000
QA_PROMPT_REVISION = 1

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


def _install_shutdown_handlers():
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, _signal_handler)
    return previous


def _restore_shutdown_handlers(previous):
    for sig, handler in previous.items():
        signal.signal(sig, handler)


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


def _read_qa_input_chunks(input_path, input_format):
    """Return ``(text, provenance)`` chunks from summaries or RAG JSONL."""
    if input_format == "summaries":
        with open(input_path, "r", encoding="utf-8") as source:
            text = source.read()
        return [(chunk, {}) for chunk in _split_into_chunks(text)]

    chunks = []
    with open(input_path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid chunk JSON on line {line_number}: {exc}"
                ) from exc
            text = item.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            provenance = {
                "source_chunk_id": item.get("id"),
                "source_document_id": item.get("document_id"),
                "source_document": item.get("source_path"),
                "source_pages": item.get("pages", []),
                "source_block_ids": item.get("block_ids", []),
                "source_heading_path": item.get("heading_path", []),
            }
            chunks.append((text, provenance))
    return chunks


def _is_junk_chunk(text, min_alpha_ratio=0.30, min_alpha_chars=40, absolute_alpha_threshold=150):
    """Return True if the chunk is mostly punctuation/whitespace garbage.

    A chunk is considered junk when:
    - It contains fewer than *min_alpha_chars* letters/digits total, OR
    - Less than *min_alpha_ratio* of its characters are alphanumeric AND
      it contains fewer than *absolute_alpha_threshold* letters/digits total.
    """
    if not text or not text.strip():
        return True
    total = len(text)
    alpha_count = sum(1 for c in text if c.isalnum())
    # If the chunk has a substantial absolute amount of readable text, it is not junk
    if alpha_count >= absolute_alpha_threshold:
        return False
    if alpha_count < min_alpha_chars:
        return True
    if alpha_count / total < min_alpha_ratio:
        return True
    return False


def _is_junk_qa(qa, min_alpha_ratio=0.40, min_question_len=10, min_answer_len=10, absolute_alpha_threshold=50):
    """Return True if a Q&A pair contains junk/garbage content.

    Detects:
    - Questions or answers that are mostly punctuation/whitespace
    - Very short or empty questions/answers
    - Repetitive character patterns (OCR artifacts like '........' or '* * * *')
    """
    q = qa.get("question", "").strip()
    a = qa.get("answer", "").strip()

    # Too short to be useful
    if len(q) < min_question_len or len(a) < min_answer_len:
        return True

    for text in (q, a):
        total = len(text)
        if total == 0:
            return True
        alpha_count = sum(1 for c in text if c.isalnum())
        # If it has a significant number of alphanumeric chars, it's not junk
        if alpha_count >= absolute_alpha_threshold:
            continue
        # Mostly non-alphanumeric
        if alpha_count / total < min_alpha_ratio:
            return True
        # Repetitive single-character pattern (e.g. "............", "* * * * *")
        unique_chars = set(text.replace(" ", ""))
        if len(unique_chars) <= 3 and total > 20:
            return True

    return False

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
            elif ord(ch) < 32:
                result.append(f'\\u{ord(ch):04x}')
            else:
                result.append(ch)
        else:
            result.append(ch)
        
        i += 1
    
    return ''.join(result)


def _extract_balanced_json_array(content):
    """Return the first top-level JSON array, ignoring brackets inside strings."""
    start = content.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    i = start
    while i < len(content):
        ch = content[i]
        if ch == "\\" and in_str:
            i += 2
            continue
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return content[start : i + 1]
        i += 1
    return None


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
    
    # Strategy 2: First balanced [...] array (avoids swallowing later brackets)
    json_str = _extract_balanced_json_array(content)
    if json_str:
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


def _try_parse_enrichment_json(content):
    """Try multiple strategies to extract a JSON object with 'input' and 'output' keys from LLM output."""

    def _attempt_parse(json_str):
        """Try parsing with sanitization and invalid-escape recovery."""
        # First try with sanitization
        sanitized = _sanitize_json_string(json_str)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass
        # Fallback: strip invalid backslash escapes (e.g. \S, \x without valid hex, etc.)
        cleaned = re.sub(r'\\(?!["\\\\bfnrtu])', r'\\\\', sanitized)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Fallback 2: aggressively strip ALL backslash sequences that aren't standard JSON escapes
        aggressive = re.sub(r'\\(?!["\\\\bfnrtu/])', '', sanitized)
        try:
            return json.loads(aggressive)
        except json.JSONDecodeError:
            pass
        return None

    def _attempt_parse_truncated(json_str):
        """Try to repair and parse truncated JSON by closing open structures."""
        # Try the normal parse first
        result = _attempt_parse(json_str)
        if result is not None:
            return result

        # Attempt to close truncated JSON: track open braces/brackets/strings
        sanitized = _sanitize_json_string(json_str)
        # Strip trailing incomplete string values (e.g. cut mid-sentence)
        # Try progressively trimming from the end to find a parseable prefix
        for trim_target in ['\n', '.', ',', ' ']:
            last_pos = sanitized.rfind(trim_target)
            while last_pos > len(sanitized) // 2:
                candidate = sanitized[:last_pos]
                # Close any open string
                if candidate.count('"') % 2 != 0:
                    candidate += '"'
                # Close open braces/brackets
                open_braces = candidate.count('{') - candidate.count('}')
                open_brackets = candidate.count('[') - candidate.count(']')
                candidate += ']' * max(0, open_brackets)
                candidate += '}' * max(0, open_braces)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                # Also try with aggressive escape stripping
                aggressive = re.sub(r'\\(?!["\\\\bfnrtu/])', '', candidate)
                try:
                    return json.loads(aggressive)
                except json.JSONDecodeError:
                    pass
                last_pos = sanitized.rfind(trim_target, 0, last_pos)
        return None

    def _validate_enrichment(result):
        """Check that a parsed dict has usable 'input' and 'output' keys."""
        if not isinstance(result, dict):
            return None
        if "input" in result and "output" in result:
            return result
        # Sometimes the LLM nests the data one level deep (e.g. {"result": {"input": ..., "output": ...}})
        for v in result.values():
            if isinstance(v, dict) and "input" in v and "output" in v:
                return v
        return None

    # Strategy 1: Extract JSON from markdown code block ```json ... ``` (non-greedy for inner braces)
    code_block_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
    if code_block_match:
        result = _attempt_parse(code_block_match.group(1))
        validated = _validate_enrichment(result)
        if validated:
            return validated

    # Strategy 1b: Strip opening ```json fence when closing ``` is missing (truncated response)
    fence_open_match = re.search(r'```(?:json)?\s*(\{.*)', content, re.DOTALL)
    if fence_open_match:
        stripped = fence_open_match.group(1).rstrip('`').rstrip()
        result = _attempt_parse(stripped)
        validated = _validate_enrichment(result)
        if validated:
            return validated
        # Try truncated repair on the fence-stripped content
        result = _attempt_parse_truncated(stripped)
        validated = _validate_enrichment(result)
        if validated:
            return validated

    # Strategy 2: Find outermost { ... } with balanced brace matching (skip braces inside strings)
    start = content.find('{')
    if start != -1:
        depth = 0
        end = start
        in_str = False
        i = start
        while i < len(content):
            ch = content[i]
            if ch == '\\' and in_str:
                i += 2  # skip escaped char
                continue
            if ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            i += 1
        if end > start:
            result = _attempt_parse(content[start:end])
            validated = _validate_enrichment(result)
            if validated:
                return validated

    # Strategy 3: Simple first-{-to-last-} extraction (fallback for nested issues)
    last = content.rfind('}')
    if start != -1 and last >= start:
        result = _attempt_parse(content[start:last + 1])
        validated = _validate_enrichment(result)
        if validated:
            return validated

    # Strategy 4: Regex extraction of individual fields when JSON structure is broken
    # Try to find "input" and "output" values even if the overall JSON is malformed
    input_match = re.search(r'"input"\s*:\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
    output_match = re.search(r'"output"\s*:\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
    if input_match and output_match:
        try:
            inp = input_match.group(1).replace('\\n', '\n').replace('\\t', ' ')
            out = output_match.group(1).replace('\\n', '\n').replace('\\t', ' ')
            if inp.strip() and out.strip():
                return {"input": inp, "output": out}
        except Exception:
            pass

    # Strategy 5: "output" might be a nested JSON object rather than a string
    if input_match:
        output_obj_match = re.search(r'"output"\s*:\s*(\{.*)', content, re.DOTALL)
        if output_obj_match:
            obj_str = output_obj_match.group(1)
            # Find balanced braces
            depth = 0
            for i, ch in enumerate(obj_str):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        result = _attempt_parse(obj_str[:i+1])
                        if isinstance(result, dict):
                            inp = input_match.group(1).replace('\\n', '\n').replace('\\t', ' ')
                            return {"input": inp, "output": result}
                        break

    # Strategy 6: "input" is a nested JSON object (not a string), "output" may also be nested
    input_obj_match = re.search(r'"input"\s*:\s*(\{)', content, re.DOTALL)
    if input_obj_match:
        # Extract balanced input object
        obj_start = input_obj_match.start(1)
        depth = 0
        in_s = False
        inp_end = obj_start
        j = obj_start
        while j < len(content):
            ch = content[j]
            if ch == '\\' and in_s:
                j += 2
                continue
            if ch == '"':
                in_s = not in_s
            elif not in_s:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        inp_end = j + 1
                        break
            j += 1
        if inp_end > obj_start:
            inp_result = _attempt_parse(content[obj_start:inp_end])
            if isinstance(inp_result, dict):
                # Now find "output" after input
                remainder = content[inp_end:]
                out_obj_match = re.search(r'"output"\s*:\s*(\{)', remainder, re.DOTALL)
                if out_obj_match:
                    out_start = out_obj_match.start(1)
                    depth = 0
                    in_s = False
                    out_end = out_start
                    k = out_start
                    while k < len(remainder):
                        ch = remainder[k]
                        if ch == '\\' and in_s:
                            k += 2
                            continue
                        if ch == '"':
                            in_s = not in_s
                        elif not in_s:
                            if ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    out_end = k + 1
                                    break
                        k += 1
                    if out_end > out_start:
                        out_result = _attempt_parse(remainder[out_start:out_end])
                        if out_result is not None:
                            return {"input": inp_result, "output": out_result}

    # Strategy 7: Last resort — truncated JSON repair on the full content from first {
    if start != -1:
        result = _attempt_parse_truncated(content[start:])
        validated = _validate_enrichment(result)
        if validated:
            return validated

    return None


def generate_qa_from_text(text, model_name="gemma3:4b-it-q4_K_M", source_file="N/A", llm_client=None):
    if llm_client is None:
        llm_client = LLMClient(provider="ollama")

    prompt = f"""
You are an expert AI dataset creator specializing in generating high-quality training data. Based on the following document, generate a list of high-quality Question/Answer pairs for fine-tuning an AI model. Prioritize specific, factual, and technical questions but also include conceptual questions when the content warrants it.

CRITICAL RULES:
1. PRIORITIZE highly specific, detailed, and factual questions that test real-world knowledge or technical details. Include concrete details such as names, command syntax, protocol specifics, tools, values, formulas, configuration parameters, registry keys, API calls, or technique names when available in the text.
2. Conceptual questions are acceptable but should remain specific and non-trivial. Avoid overly generic questions like "What is this about?" or "Why is this important?".
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
    r'\b(?:this|the)\s+(?:book|document|text|chapter|module|section|manual|guide|handbook|course|material|presentation|lecture)\b'
    r'|according\s+to\s+(?:the|this)\s+(?:book|document|text|chapter|author|manual|guide)'
    r'|as\s+(?:described|explained|stated|mentioned|discussed|noted|outlined|covered|presented|defined|highlighted)\s+in\s+(?:the|this)\s+(?:book|document|text|chapter|manual|guide)'
    r'|\bthe\s+author\s+says\b'
    r'|\bin\s+(?:chapter|module|section)\s+\d\b'
    r'|\btarget\s+audience\b'
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
    """Compute a reproducible MinHash signature for a set of shingles.

    Python's built-in string hash is randomized between processes, which made
    the LSH candidate set vary between runs. BLAKE2 creates stable base hashes;
    SplitMix64-style mixing cheaply derives deterministic permutations.
    """
    mask = 0xFFFFFFFFFFFFFFFF
    base_hashes = [
        int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for shingle in shingles
    ]
    signature = []
    for i in range(num_hashes):
        min_hash = float('inf')
        seed = (0x9E3779B97F4A7C15 * (i + 1)) & mask
        for base_hash in base_hashes:
            h = (base_hash + seed) & mask
            h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & mask
            h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & mask
            h ^= h >> 31
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
        payload = b"".join(value.to_bytes(8, "big") for value in band)
        bands.append(
            int.from_bytes(
                hashlib.blake2b(payload, digest_size=8).digest(),
                "big",
            )
        )
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

def _load_already_processed(raw_path, pipeline_fingerprint):
    """Read the raw JSONL file and return chunk-level resume state.

    Returns:
        dict[str, dict[int, str | None]]: source -> chunk index -> content hash.
        Legacy entries without hashes are loaded but are not trusted for skips.
    """
    done = defaultdict(dict)
    if not os.path.exists(raw_path):
        return done
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("_pipeline_fingerprint") != pipeline_fingerprint:
                    continue
                src = obj.get("_source_file", "")
                if src:
                    chunk_idx = obj.get("_chunk_idx", -1)
                    content_hash = (
                        obj.get("_input_sha256")
                        if chunk_idx == -1
                        else obj.get("_chunk_hash")
                    )
                    done[src][chunk_idx] = content_hash
            except json.JSONDecodeError:
                continue
    return done


def _load_raw_pairs(
    raw_path,
    current_input_hashes=None,
    pipeline_fingerprint=None,
):
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
                if (
                    pipeline_fingerprint is not None
                    and obj.get("_pipeline_fingerprint") != pipeline_fingerprint
                ):
                    continue
                if "question" in obj and "answer" in obj:
                    source_id = obj.get("_source_file")
                    recorded_input_hash = obj.get("_input_sha256")
                    if current_input_hashes is not None and source_id:
                        if source_id not in current_input_hashes:
                            continue
                        if (
                            recorded_input_hash
                            and recorded_input_hash != current_input_hashes[source_id]
                        ):
                            continue
                    pair = {
                        "question": obj["question"],
                        "answer": obj["answer"],
                    }
                    pair.update(
                        {
                            key: value
                            for key, value in obj.items()
                            if key.startswith("_source_")
                        }
                    )
                    pairs.append(pair)
            except json.JSONDecodeError:
                continue
    return pairs


def generate_qa_dataset(
    source_dir,
    dest_dir,
    model_name,
    llm_client=None,
    num_threads=1,
    force=False,
    enrich=False,
    enrich_ratio=0.3,
    input_format="auto",
    enrich_domain="cyber",
):
    """Generate a Q&A dataset, installing shutdown handlers only for this run."""
    global _shutdown_requested
    _shutdown_requested = False
    previous_handlers = _install_shutdown_handlers()
    try:
        return _run_qa_dataset(
            source_dir,
            dest_dir,
            model_name,
            llm_client=llm_client,
            num_threads=num_threads,
            force=force,
            enrich=enrich,
            enrich_ratio=enrich_ratio,
            input_format=input_format,
            enrich_domain=enrich_domain,
        )
    finally:
        _restore_shutdown_handlers(previous_handlers)


def _run_qa_dataset(
    source_dir,
    dest_dir,
    model_name,
    llm_client=None,
    num_threads=1,
    force=False,
    enrich=False,
    enrich_ratio=0.3,
    input_format="auto",
    enrich_domain="cyber",
):
    if llm_client is None:
        llm_client = LLMClient(provider="ollama")

    qa_pipeline_config = {
        "qa_prompt_revision": QA_PROMPT_REVISION,
        "model": model_name,
        "llm_generation": (
            llm_client.generation_config()
            if hasattr(llm_client, "generation_config")
            else {
                "provider": getattr(llm_client, "provider", "unknown"),
            }
        ),
    }
    qa_pipeline_fingerprint = stable_fingerprint(qa_pipeline_config)

    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist. Creating it now...")
        os.makedirs(source_dir, exist_ok=True)

    os.makedirs(dest_dir, exist_ok=True)
    
    if input_format not in {"auto", "summaries", "chunks"}:
        raise ValueError("input_format must be auto, summaries, or chunks")

    markdown_files = []
    chunk_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            path = os.path.join(root, file)
            if file.lower().endswith(".chunks.jsonl"):
                chunk_files.append(path)
            elif file.lower().endswith(".md"):
                markdown_files.append(path)

    if input_format == "auto":
        resolved_input_format = "chunks" if chunk_files else "summaries"
    else:
        resolved_input_format = input_format
    files_to_process = (
        chunk_files if resolved_input_format == "chunks" else markdown_files
    )
    files_to_process.sort()

    raw_filename = (
        "dataset_qa_raw_chunks.jsonl"
        if resolved_input_format == "chunks"
        else "dataset_qa_raw.jsonl"
    )
    raw_jsonl_path = os.path.join(dest_dir, raw_filename)
    
    if force and os.path.exists(raw_jsonl_path):
        try:
            os.remove(raw_jsonl_path)
            print(f"Force option enabled: resetting raw Q&A data at {raw_jsonl_path}")
        except Exception as e:
            print(f"Warning: Could not remove {raw_jsonl_path}: {e}")
    
    if not files_to_process:
        expected = "*.chunks.jsonl" if resolved_input_format == "chunks" else "*.md"
        print(f"No {expected} files found in '{source_dir}'.")
        return
    
    # Check which files/chunks were already processed (resume support)
    already_done = (
        _load_already_processed(raw_jsonl_path, qa_pipeline_fingerprint)
        if not force
        else defaultdict(dict)
    )
    # A file-level sentinel is valid only while the input content hash matches.
    input_hashes = {
        relative_source_id(path, source_dir): sha256_file(path)
        for path in files_to_process
    }
    raw_has_other_state = (
        os.path.exists(raw_jsonl_path)
        and os.path.getsize(raw_jsonl_path) > 0
        and not already_done
    )
    resume_state_changed = raw_has_other_state or bool(
        set(already_done) - set(input_hashes)
    )
    for source_id, source_hash in input_hashes.items():
        recorded_hash = already_done.get(source_id, {}).get(-1)
        if recorded_hash and recorded_hash != source_hash:
            resume_state_changed = True
    fully_done_files = set()
    for path in files_to_process:
        source_id = relative_source_id(path, source_dir)
        expected_hash = already_done.get(source_id, {}).get(-1)
        if expected_hash and expected_hash == input_hashes[source_id]:
            fully_done_files.add(source_id)
    partially_done_files = {f for f in already_done if f not in fully_done_files}
    if already_done:
        total_before = len(files_to_process)
        # Remove fully-processed files; keep partially-processed ones for chunk-level resume
        files_to_process = [
            f for f in files_to_process
            if relative_source_id(f, source_dir) not in fully_done_files
        ]
        skipped = total_before - len(files_to_process)
        partial_msg = f" ({len(partially_done_files)} partially done — will resume at chunk level)" if partially_done_files else ""
        print(f"Resuming: {skipped} files already processed, {len(files_to_process)} remaining.{partial_msg}")
        
    dataset_json_path = os.path.join(dest_dir, "dataset_qa.json")
    dataset_md_path = os.path.join(dest_dir, "dataset_qa.md")
    
    if not force and len(files_to_process) == 0 and not resume_state_changed:
        if os.path.exists(dataset_json_path) and os.path.exists(dataset_md_path):
            if enrich and enrich_ratio > 0:
                print("Resuming: Q&A dataset is already complete and all files have been processed. Jumping directly to enrichment (Phase 5).")
            else:
                print("Resuming: Q&A dataset is already complete and all files have been processed. Skipping completely.")
                llm_client.unload_model(model_name)
                return
    
    # Count existing pairs without loading them all into memory
    total_saved = 0
    if os.path.exists(raw_jsonl_path):
        with open(raw_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source_id = item.get("_source_file")
                if (
                    "question" in item
                    and "answer" in item
                    and item.get("_pipeline_fingerprint")
                    == qa_pipeline_fingerprint
                    and source_id in input_hashes
                    and item.get("_input_sha256") == input_hashes[source_id]
                ):
                    total_saved += 1

    total_saved_before = total_saved  # snapshot to detect new pairs added by Phase 1
        
    if files_to_process:
        print(
            f"Processing {len(files_to_process)} {resolved_input_format} files "
            "for Q&A generation..."
        )
        if num_threads > 1:
            print(f"Using {num_threads} threads for parallel chunk processing.")
        
        failed_files = 0
        raw_write_lock = threading.Lock()
        pbar = tqdm(files_to_process, desc="Generating Q&A", unit="file")
             
        for input_path in pbar:
            filename = os.path.basename(input_path)
            source_id = relative_source_id(input_path, source_dir)
            pbar.set_postfix({"file": filename[:20], "q_saved": total_saved})
            
            try:
                input_chunks = _read_qa_input_chunks(
                    input_path, resolved_input_format
                )

                if not input_chunks:
                    log_error(input_path, "File is empty.")
                    continue
                chunks = [item[0] for item in input_chunks]
                provenance_by_idx = {
                    index: item[1] for index, item in enumerate(input_chunks)
                }
                chunk_hashes = {
                    index: hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    for index, chunk in enumerate(chunks)
                }
                input_sha256 = sha256_file(input_path)
                
                # Determine which chunks were already processed (chunk-level resume)
                done_chunks = already_done.get(source_id, {})
                if done_chunks and any(
                    done_chunks.get(index) != chunk_hash
                    for index, chunk_hash in chunk_hashes.items()
                ):
                    resume_state_changed = True
                
                file_count = 0
                filtered_count = 0
                chunk_errors = 0
                junk_chunks = 0
                processed_chunks = 0
                already_done_count = 0

                def _flush_chunk_pairs(chunk_pairs, chunk_idx):
                    """Write Q&A pairs from one chunk to disk immediately."""
                    nonlocal file_count, filtered_count, total_saved
                    if not isinstance(chunk_pairs, list):
                        return
                    with raw_write_lock:
                        with open(raw_jsonl_path, 'a', encoding='utf-8') as f_raw:
                            for qa in chunk_pairs:
                                if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                                    if _references_source_material(qa):
                                        filtered_count += 1
                                        continue
                                    record = {
                                        "question": qa['question'],
                                        "answer": qa['answer'],
                                        "_source_file": source_id,
                                        "_chunk_idx": chunk_idx,
                                        "_chunk_hash": chunk_hashes[chunk_idx],
                                        "_input_sha256": input_sha256,
                                        "_pipeline_fingerprint": qa_pipeline_fingerprint,
                                    }
                                    provenance = provenance_by_idx.get(chunk_idx, {})
                                    for key, value in provenance.items():
                                        if value is not None:
                                            record[f"_{key}"] = value
                                    f_raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                                    file_count += 1
                                    total_saved += 1

                if num_threads > 1 and len(chunks) > 1:
                    # Parallel chunk processing — flush per chunk as results arrive
                    futures = {}
                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                        for chunk_idx, chunk in enumerate(chunks):
                            if not chunk:
                                continue
                            if done_chunks.get(chunk_idx) == chunk_hashes[chunk_idx]:
                                already_done_count += 1
                                continue
                            if _is_junk_chunk(chunk):
                                junk_chunks += 1
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
                            processed_chunks += 1
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
                        if done_chunks.get(chunk_idx) == chunk_hashes[chunk_idx]:
                            already_done_count += 1
                            continue
                        if _is_junk_chunk(chunk):
                            junk_chunks += 1
                            continue
                        if len(chunks) > 1:
                            pbar.set_postfix({"file": filename[:20], "chunk": f"{chunk_idx+1}/{len(chunks)}", "q_saved": total_saved})
                        
                        processed_chunks += 1
                        try:
                            chunk_pairs = generate_qa_from_text(chunk, model_name=model_name, source_file=input_path, llm_client=llm_client)
                            _flush_chunk_pairs(chunk_pairs, chunk_idx)
                        except Exception as e:
                            chunk_errors += 1
                            log_error(input_path, f"Chunk {chunk_idx}: {e}")
                            tqdm.write(f"  [{filename}] chunk {chunk_idx} error: {e}")
                            continue

                # End-of-file reporting and sentinel logic
                if processed_chunks == 0 and file_count == 0:
                    if already_done_count > 0 and junk_chunks == 0:
                        # All chunks were already processed on a previous run — write sentinel silently
                        with open(raw_jsonl_path, 'a', encoding='utf-8') as f_raw:
                            sentinel = {
                                "_source_file": source_id,
                                "_chunk_idx": -1,
                                "_input_sha256": input_sha256,
                                "_pipeline_fingerprint": qa_pipeline_fingerprint,
                                "_skipped": False,
                            }
                            f_raw.write(json.dumps(sentinel, ensure_ascii=False) + "\n")
                    else:
                        # No chunks sent to LLM (all junk/empty) — mark as skipped
                        with open(raw_jsonl_path, 'a', encoding='utf-8') as f_raw:
                            sentinel = {
                                "_source_file": source_id,
                                "_chunk_idx": -1,
                                "_input_sha256": input_sha256,
                                "_pipeline_fingerprint": qa_pipeline_fingerprint,
                                "_skipped": True,
                            }
                            f_raw.write(json.dumps(sentinel, ensure_ascii=False) + "\n")
                        if junk_chunks > 0:
                            tqdm.write(f"  [{filename}] Skipped — all {junk_chunks} chunks are junk/garbage.")
                            log_error(input_path, f"Skipped — all {junk_chunks} chunks are junk/garbage content.")
                        else:
                            tqdm.write(f"  [{filename}] Skipped — no processable chunks.")
                elif file_count == 0 and chunk_errors == 0:
                    failed_files += 1
                    log_error(input_path, "No valid Q&A pairs extracted.")
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
        new_pairs_count = total_saved - total_saved_before
    else:
        new_pairs_count = 0
        print("All files already processed. Skipping to deduplication.")

    # Fast-path: if Phase 1 added no new pairs and the final dataset already exists,
    # skip the expensive phases 2-4 (filter + dedup + save) entirely.
    dataset_json_path = os.path.join(dest_dir, "dataset_qa.json")
    dataset_md_path = os.path.join(dest_dir, "dataset_qa.md")

    if (
        new_pairs_count == 0
        and not resume_state_changed
        and os.path.exists(dataset_json_path)
        and os.path.exists(dataset_md_path)
    ):
        print("\nNo new Q&A pairs generated. Final dataset already exists — skipping phases 2-4.")
        # Jump directly to enrichment (Phase 5) if requested
        if enrich and enrich_ratio > 0:
            try:
                with open(dataset_json_path, 'r', encoding='utf-8') as f_json:
                    alpaca_format = json.load(f_json)
                print(f"Loaded {len(alpaca_format)} entries from existing dataset for enrichment.")
                dataset_enriched_path = os.path.join(dest_dir, "dataset_qa_enriched.json")
                enriched_pairs = _enrich_after_dedup(list(alpaca_format), ratio=enrich_ratio, model_name=model_name, llm_client=llm_client, output_path=dataset_enriched_path, domain=enrich_domain)
                atomic_write_text(
                    dataset_enriched_path,
                    json.dumps(enriched_pairs, ensure_ascii=False, indent=2) + "\n",
                )
                enriched_count = sum(1 for e in enriched_pairs if isinstance(e.get("_meta"), dict) and e["_meta"].get("enriched"))
                print(f"Saved enriched dataset ({enriched_count} entries enriched) to: {dataset_enriched_path}")
                clean_dataset(dataset_json_path)
                cleaned_enriched = clean_dataset(dataset_enriched_path)
                if cleaned_enriched:
                    prepare_hf_dataset(cleaned_enriched)
            except (json.JSONDecodeError, Exception) as e:
                print(f"  [WARNING] Could not load existing dataset for enrichment: {e}")
                print("  Falling through to full phases 2-4...")
            else:
                llm_client.unload_model(model_name)
                return
        else:
            cleaned_json = clean_dataset(dataset_json_path)
            if cleaned_json:
                prepare_hf_dataset(cleaned_json)
            llm_client.unload_model(model_name)
            return
    
    # Load all raw pairs from disk for deduplication
    all_qa_pairs = _load_raw_pairs(
        raw_jsonl_path,
        current_input_hashes=input_hashes,
        pipeline_fingerprint=qa_pipeline_fingerprint,
    )
    
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

    # Phase 2b: Filter out junk Q&A pairs (garbage content from OCR artifacts)
    before_junk = len(clean_pairs)
    clean_pairs = [qa for qa in clean_pairs if not _is_junk_qa(qa)]
    filtered_junk = before_junk - len(clean_pairs)
    if filtered_junk:
        print(f"  Removed {filtered_junk} junk pairs (garbage/punctuation content).")

    # Phase 3: Deduplication
    print(f"Phase 3: Removing duplicates from {len(clean_pairs)} pairs (similarity threshold 85%)...")
    unique_qa_pairs = deduplicate_qa(clean_pairs, threshold=0.85)
    
    print("Phase 4: Saving final clean files...")
    
    # Standard Alpaca json format (built before the with-block so it's available for enrichment)
    alpaca_format = []
    for qa in unique_qa_pairs:
        provenance = {
            key[len("_source_") :]: value
            for key, value in qa.items()
            if key.startswith("_source_")
        }
        entry = {
            "instruction": qa["question"],
            "input": "",
            "output": qa["answer"],
        }
        if provenance:
            entry["_meta"] = {"source": provenance}
        alpaca_format.append(entry)

    atomic_write_text(
        dataset_json_path,
        json.dumps(alpaca_format, ensure_ascii=False, indent=2) + "\n",
    )
    markdown_parts = ["# QA Dataset\n\n"]
    for qa in unique_qa_pairs:
        markdown_parts.append(f"**Q: {qa['question']}**\n\n")
        markdown_parts.append(f"**A:** {qa['answer']}\n\n")
        markdown_parts.append("---\n\n")
    atomic_write_text(dataset_md_path, "".join(markdown_parts))

    print(f"\nQ&A Generation Complete! Generated {len(unique_qa_pairs)} unique Q&A pairs (Removed {len(all_qa_pairs) - len(unique_qa_pairs)} duplicates).")
    print(f"Saved JSON dataset to: {dataset_json_path}")
    print(f"Saved Markdown readable dataset to: {dataset_md_path}")
    print(f"Raw data preserved in: {raw_jsonl_path}")

    # Clean standard dataset
    cleaned_json = clean_dataset(dataset_json_path)

    # Phase 5: Post-deduplication enrichment (only in full pipeline mode)
    if enrich and enrich_ratio > 0:
        dataset_enriched_path = os.path.join(dest_dir, "dataset_qa_enriched.json")
        enriched_pairs = _enrich_after_dedup(list(alpaca_format), ratio=enrich_ratio, model_name=model_name, llm_client=llm_client, output_path=dataset_enriched_path, domain=enrich_domain)
        # Final save (also done incrementally inside the function for crash safety)
        atomic_write_text(
            dataset_enriched_path,
            json.dumps(enriched_pairs, ensure_ascii=False, indent=2) + "\n",
        )
        enriched_count = sum(1 for e in enriched_pairs if isinstance(e.get("_meta"), dict) and e["_meta"].get("enriched"))
        print(f"Saved enriched dataset ({enriched_count} entries enriched) to: {dataset_enriched_path}")
        cleaned_enriched = clean_dataset(dataset_enriched_path)
        if cleaned_enriched:
            prepare_hf_dataset(cleaned_enriched)
    else:
        if cleaned_json:
            prepare_hf_dataset(cleaned_json)
        if enrich:
            print("\nPhase 5: Enrichment skipped (enrich-ratio is 0).")



    # Unload model from VRAM (only for Ollama)
    llm_client.unload_model(model_name)


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


