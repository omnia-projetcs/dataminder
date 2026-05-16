import os
import argparse
from datetime import datetime
from extractor import extract_text
from summarizer import summarize_text
from tqdm import tqdm

def process_documents(source_dir, dest_dir, model_name):
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Scanning '{source_dir}' recursively...")
    
    files_to_process = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.txt', '.md', '.docx', '.pdf', '.cbz', '.cbr', '.doc', '.pptx', '.xls', '.xlsx', '.html', '.htm', '.chm']:
                files_to_process.append(os.path.join(root, file))
    
    if not files_to_process:
        print(f"No supported documents (txt, md, pdf, doc, docx, pptx, xls, xlsx, cbz, cbr, html, chm) found in '{source_dir}'.")
        return
        
    print(f"Found {len(files_to_process)} documents to process.")
    
    pbar = tqdm(files_to_process, desc="Processing", unit="doc")
    for input_path in pbar:
        filename = os.path.basename(input_path)
        base_name, _ = os.path.splitext(filename)
        
        # Output filename
        output_filename = f"{base_name}.md"
        output_path = os.path.join(dest_dir, output_filename)
            
        pbar.set_postfix({"file": filename[:20], "step": "Extracting text"})
        text = extract_text(input_path)
        
        if not text.strip():
            tqdm.write(f"[{filename}] No text could be extracted. Skipping.")
            continue
            
        pbar.set_postfix({"file": filename[:20], "step": "AI Summarizing"})
        summary_md = summarize_text(text, model_name=model_name)
        
        pbar.set_postfix({"file": filename[:20], "step": "Saving"})
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_md)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and summarize documents.")
    parser.add_argument("--source", default="source", help="Source directory containing the documents to process (default: source).")
    parser.add_argument("--dest", default="destination", help="Destination directory for the Markdown summaries (default: destination).")
    parser.add_argument("--model", default="ministral-3:8b", help="Ollama model to use (default: ministral-3:8b).")
    
    args = parser.parse_args()
    
    print("--- Starting document processing ---")
    process_documents(args.source, args.dest, args.model)
