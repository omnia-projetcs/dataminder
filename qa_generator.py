import os
import json
import re
from tqdm import tqdm
import ollama
import difflib
import datetime

def log_error(filepath, error_msg):
    with open("error.log", "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] FILE: {filepath} | QA GENERATION ERROR: {error_msg}\n")

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


def generate_qa_from_text(text, model_name="gemma3:4b", source_file="N/A"):
    prompt = f"""
You are an expert AI dataset creator specializing in technical cybersecurity and IT training data. Based on the following document, generate a list of high-quality Question/Answer pairs for fine-tuning an AI model. Prioritize technical, hands-on questions but also include conceptual questions when the content warrants it.

CRITICAL RULES:
1. PRIORITIZE highly technical and specific questions that test real-world, hands-on knowledge. Include concrete details such as tool names, command syntax, protocol specifics, CVE references, configuration parameters, registry keys, API calls, or attack technique names when available.
2. Conceptual questions are acceptable but should remain specific and non-trivial. Avoid overly generic questions like "What is cybersecurity?" or "Why is security important?".
3. Answers must be self-contained, precise, and technically accurate. Include specific values, commands, paths, or configurations when relevant.
4. Keep answers in plain text. Do NOT include code blocks, markdown formatting, or newlines inside answers.
5. NEVER reference the source document, book, author, chapter, or publication. Write questions and answers as standalone technical knowledge.
6. Provide the output STRICTLY as a JSON array of objects. No other text.
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
        response = ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': prompt}
        ])
        
        content = response['message']['content']
        
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
        tqdm.write(f"  [ERROR] Ollama call failed: {e}")
        log_error(source_file, f"Ollama exception: {e}")
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

def deduplicate_qa(qa_list, threshold=0.85):
    unique_qa = []
    for qa in tqdm(qa_list, desc="Deduplicating", unit="pair"):
        is_duplicate = False
        q_text = qa.get("question", "").lower()
        for u_qa in unique_qa:
            u_q_text = u_qa.get("question", "").lower()
            # SequenceMatcher ratio > 0.85 means highly similar questions
            if difflib.SequenceMatcher(None, q_text, u_q_text).ratio() > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_qa.append(qa)
    return unique_qa

def _load_already_processed(raw_path):
    """Read the raw JSONL file and return a set of source filenames already processed."""
    done = set()
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
                    done.add(src)
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


def generate_qa_dataset(source_dir, dest_dir, model_name):
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist. Creating it now...")
        os.makedirs(source_dir, exist_ok=True)

    os.makedirs(dest_dir, exist_ok=True)
    
    raw_jsonl_path = os.path.join(dest_dir, "dataset_qa_raw.jsonl")
    
    files_to_process = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.md'):
                files_to_process.append(os.path.join(root, file))
                
    if not files_to_process:
        print(f"No Markdown (.md) files found in '{source_dir}'.")
        return
    
    # Check which files were already processed (resume support)
    already_done = _load_already_processed(raw_jsonl_path)
    if already_done:
        total_before = len(files_to_process)
        files_to_process = [f for f in files_to_process if os.path.basename(f) not in already_done]
        print(f"Resuming: {total_before - len(files_to_process)} files already processed, {len(files_to_process)} remaining.")
    
    total_saved = len(_load_raw_pairs(raw_jsonl_path))
        
    if files_to_process:
        print(f"Processing {len(files_to_process)} Markdown files for Q&A generation...")
        
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
                    
                # Limit text size if it's too huge to prevent context blowup
                text = text[:20000] 
                    
                qa_pairs = generate_qa_from_text(text, model_name=model_name, source_file=input_path)
                
                if isinstance(qa_pairs, list):
                    file_count = 0
                    filtered_count = 0
                    # Write each valid pair immediately to the raw JSONL file
                    with open(raw_jsonl_path, 'a', encoding='utf-8') as f_raw:
                        for qa in qa_pairs:
                            if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                                # Filter out pairs that reference source material
                                if _references_source_material(qa):
                                    filtered_count += 1
                                    continue
                                record = {
                                    "question": qa['question'],
                                    "answer": qa['answer'],
                                    "_source_file": filename
                                }
                                f_raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                                file_count += 1
                                total_saved += 1
                    if file_count == 0:
                        failed_files += 1
                        tqdm.write(f"  [{filename}] No valid Q&A pairs extracted.")
                    else:
                        msg = f"  [{filename}] +{file_count} pairs saved"
                        if filtered_count:
                            msg += f" ({filtered_count} filtered: source refs)"
                        msg += f" (total: {total_saved})"
                        tqdm.write(msg)
                else:
                    failed_files += 1
                    log_error(input_path, "AI did not return a valid list of QA pairs.")
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
        print("Common causes: Ollama not running, model not available, or all AI responses were unparseable.")
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
    
    # Save JSON array
    with open(dataset_json_path, 'w', encoding='utf-8') as f_json:
        # Standard Alpaca json format
        alpaca_format = [{"instruction": qa["question"], "input": "", "output": qa["answer"]} for qa in unique_qa_pairs]
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
