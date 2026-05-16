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

def generate_qa_from_text(text, model_name="ministral-3:8b"):
    prompt = f"""
You are an expert AI dataset creator. Based on the following document, generate a list of high-quality Question/Answer pairs to be used for fine-tuning an AI model.

CRITICAL RULES:
1. Generate specific, detailed questions that require understanding the document. Avoid generic questions.
2. The answers must be self-contained and comprehensive.
3. Provide the output STRICTLY as a JSON array of objects. Do not write any other text or markdown block formatting.
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
        # Try to extract JSON array using regex if the LLM added markdown formatting like ```json
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            return json.loads(content)
            
    except Exception as e:
        # Silently fail for a single document, return empty list
        return []

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

def generate_qa_dataset(source_dir, dest_dir, model_name):
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist. Creating it now...")
        os.makedirs(source_dir, exist_ok=True)

    os.makedirs(dest_dir, exist_ok=True)
    
    files_to_process = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.md'):
                files_to_process.append(os.path.join(root, file))
                
    if not files_to_process:
        print(f"No Markdown (.md) files found in '{source_dir}'.")
        return
        
    print(f"Found {len(files_to_process)} Markdown files to process for Q&A generation.")
    
    all_qa_pairs = []
    pbar = tqdm(files_to_process, desc="Generating Q&A", unit="file")
         
    for input_path in pbar:
        filename = os.path.basename(input_path)
        pbar.set_postfix({"file": filename[:20], "q_found": len(all_qa_pairs)})
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            if not text.strip():
                log_error(input_path, "File is empty.")
                continue
                
            # Limit text size if it's too huge to prevent context blowup
            text = text[:20000] 
                
            qa_pairs = generate_qa_from_text(text, model_name=model_name)
            
            if isinstance(qa_pairs, list):
                for qa in qa_pairs:
                    if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                        all_qa_pairs.append({
                            "question": qa['question'],
                            "answer": qa['answer']
                        })
            else:
                log_error(input_path, "AI did not return a valid list of QA pairs.")
        except Exception as e:
            log_error(input_path, f"Unexpected exception: {str(e)}")
            continue

    # Deduplication phase
    print(f"\nPhase 1 Complete. Generated {len(all_qa_pairs)} initial Q&A pairs.")
    print("Phase 2: Removing duplicates (similarity threshold 85%)...")
    unique_qa_pairs = deduplicate_qa(all_qa_pairs, threshold=0.85)
    
    dataset_json_path = os.path.join(dest_dir, "dataset_qa.json")
    dataset_md_path = os.path.join(dest_dir, "dataset_qa.md")
    
    print("Phase 3: Saving files...")
    
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
