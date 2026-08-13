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
from llm_client import LLMClient
from logger import log_error as _log_error
from document_ir import sha256_file
from processing_manifest import atomic_write_text, relative_source_id, stable_fingerprint
from dataset_export import clean_dataset, prepare_hf_dataset
from json_repair import _try_parse_json
from qa_enrichment import _enrich_after_dedup
from qa_dedup import deduplicate_qa


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
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
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
        with open(input_path, "r", encoding="utf-8-sig") as source:
            text = source.read()
        return [(chunk, {}) for chunk in _split_into_chunks(text)]

    chunks = []
    with open(input_path, "r", encoding="utf-8-sig") as source:
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

def _load_already_processed(raw_path, pipeline_fingerprint):
    """Read the raw JSONL file and return chunk-level resume state.

    Returns:
        dict[str, dict[int, str | None]]: source -> chunk index -> content hash.
        Legacy entries without hashes are loaded but are not trusted for skips.
    """
    done = defaultdict(dict)
    if not os.path.exists(raw_path):
        return done
    with open(raw_path, 'r', encoding='utf-8-sig') as f:
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
    with open(raw_path, 'r', encoding='utf-8-sig') as f:
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
        print(f"Source directory '{source_dir}' does not exist.")
        return

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
                            if _shutdown_requested:
                                for pending in futures:
                                    pending.cancel()
                            chunk_idx = futures[future]
                            completed += 1
                            processed_chunks += 1
                            pbar.set_postfix({"file": filename[:20], "chunk": f"{completed}/{len(futures)}", "q_saved": total_saved})
                            try:
                                if future.cancelled():
                                    continue
                                chunk_pairs = future.result()
                                _flush_chunk_pairs(chunk_pairs, chunk_idx)
                            except Exception as e:
                                chunk_errors += 1
                                log_error(input_path, f"Chunk {chunk_idx}: {e}")
                                tqdm.write(f"  [{filename}] chunk {chunk_idx} error: {e}")
                            if _shutdown_requested:
                                tqdm.write(f"\n  [SHUTDOWN] Stopping gracefully. Progress saved ({total_saved} pairs on disk). Resume will pick up here.")
                                pbar.close()
                                print(f"\nGraceful shutdown complete. {total_saved} total Q&A pairs saved to: {raw_jsonl_path}")
                                print("Re-run the same command to resume from where you left off.")
                                llm_client.unload_model(model_name)
                                return
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
