import os
import argparse
import ollama
from datetime import datetime
from extractor import extract_text
from summarizer import summarize_text
from tqdm import tqdm

def _unload_model(model_name):
    """Send a dummy request with keep_alive=0 to unload the model from VRAM."""
    try:
        ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': '.'}
        ], keep_alive=0)
    except Exception:
        pass

def log_error(filepath, error_msg):
    with open("error.log", "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] FILE: {filepath} | PIPELINE ERROR: {error_msg}\n")

def process_documents(source_dir, dest_dir, model_name, level=7):
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist. Creating it now...")
        os.makedirs(source_dir, exist_ok=True)

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Scanning '{source_dir}' recursively...")
    
    files_to_process = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.txt', '.md', '.docx', '.pdf', '.cbz', '.cbr', '.doc', '.pptx', '.xls', '.xlsx', '.html', '.htm', '.chm', '.epub']:
                files_to_process.append(os.path.join(root, file))
    
    if not files_to_process:
        print(f"No supported documents (txt, md, pdf, doc, docx, pptx, xls, xlsx, cbz, cbr, html, chm, epub) found in '{source_dir}'.")
        return
        
    print(f"Found {len(files_to_process)} documents to process.")
    
    pbar = tqdm(files_to_process, desc="Processing", unit="doc")
    for input_path in pbar:
        filename = os.path.basename(input_path)
        base_name, _ = os.path.splitext(filename)
        
        # Output filename
        output_filename = f"{base_name}.md"
        output_path = os.path.join(dest_dir, output_filename)
        
        try:
            # Skip if already processed
            if os.path.exists(output_path):
                continue
                
            pbar.set_postfix({"file": filename[:20], "step": "Extracting text"})
            text = extract_text(input_path)
            
            if not text.strip():
                tqdm.write(f"\n[{filename}] No text could be extracted. Skipping.")
                log_error(input_path, "No text extracted (unsupported or OCR failed).")
                continue
                
            if level == 0:
                pbar.set_postfix({"file": filename[:20], "step": "Saving Raw Text"})
                summary_md = text
            else:
                pbar.set_postfix({"file": filename[:20], "step": f"AI Summarizing (L{level})"})
                summary_md = summarize_text(text, model_name=model_name, level=level)
            
            pbar.set_postfix({"file": filename[:20], "step": "Saving"})
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary_md)
                
        except Exception as e:
            error_details = str(e)
            tqdm.write(f"\n[{filename}] Failed with error: {error_details}")
            log_error(input_path, f"Unexpected exception: {error_details}")
            continue

    # Unload model from VRAM now that processing is complete
    _unload_model(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and summarize documents.")
    parser.add_argument("--source", default="source", help="Source directory containing the documents to process (default: source).")
    parser.add_argument("--dest", default="destination", help="Destination directory for the Markdown summaries (default: destination).")
    parser.add_argument("--model", default="gemma3:12b", help="Ollama model to use (default: gemma3:12b).")
    parser.add_argument("--level", type=int, default=9, help="Summarization detail level from 1 to 10. 0 means no summarization (saves raw text). Default: 9.")
    parser.add_argument("--qa", action="store_true", help="Enable QA Dataset Generation mode (reads .md files from source and creates a dataset in dest).")
    parser.add_argument("--full", action="store_true", help="Run the full pipeline: Document summarization followed by QA Dataset generation.")
    
    args = parser.parse_args()
    
    if args.full:
        from qa_generator import generate_qa_dataset
        print("--- Starting FULL Pipeline ---")
        print("\n[Step 1/2] Document Processing")
        process_documents(args.source, args.dest, args.model, args.level)
        
        print("\n[Step 2/2] Q&A Dataset Generation")
        qa_dest = "dataresults" if args.dest == "destination" else f"{args.dest}_qa"
        generate_qa_dataset(args.dest, qa_dest, args.model)
        print("\n--- Full Pipeline Complete ---")
    elif args.qa:
        from qa_generator import generate_qa_dataset
        print("--- Starting Q&A Dataset Generation ---")
        
        # Override defaults specifically for QA mode if they haven't been manually changed
        qa_source = "destination" if args.source == "source" else args.source
        qa_dest = "dataresults" if args.dest == "destination" else args.dest
        
        generate_qa_dataset(qa_source, qa_dest, args.model)
    else:
        print("--- Starting document processing ---")
        process_documents(args.source, args.dest, args.model, args.level)
