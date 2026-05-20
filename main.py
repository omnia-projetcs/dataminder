import os
import time
import argparse
from datetime import datetime
from extractor import extract_text, configure_ocr
from summarizer import summarize_text
from llm_client import LLMClient
from tqdm import tqdm

CHUNK_SIZE = 5000

def log_error(filepath, error_msg):
    with open("error.log", "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] FILE: {filepath} | PIPELINE ERROR: {error_msg}\n")

def process_documents(source_dir, dest_dir, model_name, level=7, force=False, llm_client=None, num_threads=1, structured=False):
    if llm_client is None:
        llm_client = LLMClient(provider="ollama")

    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist. Creating it now...")
        os.makedirs(source_dir, exist_ok=True)

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Scanning '{source_dir}' recursively...")
    
    files_to_process = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.txt', '.md', '.rst', '.docx', '.pdf', '.cbz', '.cbr', '.doc', '.pptx', '.xls', '.xlsx', '.html', '.htm', '.chm', '.epub', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif']:
                files_to_process.append(os.path.join(root, file))
    
    if not files_to_process:
        print(f"No supported documents (txt, md, rst, pdf, doc, docx, pptx, xls, xlsx, cbz, cbr, html, chm, epub, png, jpg, webp, bmp, tiff) found in '{source_dir}'.")
        return
        
    # Resume: filter out already processed files unless --force
    skipped = 0
    if not force:
        filtered = []
        for fp in files_to_process:
            base_name, _ = os.path.splitext(os.path.basename(fp))
            output_path = os.path.join(dest_dir, f"{base_name}.md")
            if os.path.exists(output_path):
                skipped += 1
            else:
                filtered.append(fp)
        files_to_process = filtered
    
    if skipped:
        print(f"Resuming: {skipped} files already processed (use --force to reprocess). {len(files_to_process)} remaining.")
    
    if not files_to_process:
        print("All files already processed. Nothing to do.")
        return
    
    print(f"Processing {len(files_to_process)} documents...")
    if num_threads > 1:
        print(f"Using {num_threads} threads for parallel chunk processing.")
    
    pbar = tqdm(files_to_process, desc="Processing", unit="doc")
    for input_path in pbar:
        filename = os.path.basename(input_path)
        base_name, _ = os.path.splitext(filename)
        
        # Output filename
        output_filename = f"{base_name}.md"
        output_path = os.path.join(dest_dir, output_filename)
        
        try:
                
            pbar.set_postfix({"file": filename[:20], "step": "Extracting text"})
            text = extract_text(input_path, structured=structured)
            
            if not text.strip():
                tqdm.write(f"\n[{filename}] No text could be extracted. Skipping.")
                log_error(input_path, "No text extracted (unsupported or OCR failed).")
                continue
                
            if level == 0:
                pbar.set_postfix({"file": filename[:20], "step": "Saving Raw Text"})
                summary_md = text
            else:
                # Split text into chunks to prevent context blowup for large documents
                chunks = []
                start = 0
                while start < len(text):
                    if len(text) - start <= CHUNK_SIZE:
                        chunks.append(text[start:])
                        break
                    
                    # Try to find a paragraph break to split cleanly
                    split_point = text.rfind('\n\n', start, start + CHUNK_SIZE)
                    if split_point == -1 or split_point <= start:
                        # Fallback to newline
                        split_point = text.rfind('\n', start, start + CHUNK_SIZE)
                    if split_point == -1 or split_point <= start:
                        # Fallback to hard split
                        split_point = start + CHUNK_SIZE
                        
                    chunks.append(text[start:split_point].strip())
                    start = split_point

                if len(chunks) <= 1:
                    pbar.set_postfix({"file": filename[:20], "step": f"AI Summarizing (L{level})"})
                    t0 = time.time()
                    summary_md = summarize_text(text, model_name=model_name, level=level, llm_client=llm_client)
                    elapsed = time.time() - t0
                    pbar.set_postfix({"file": filename[:20], "step": f"Done", "time": f"{elapsed:.1f}s"})
                elif num_threads > 1:
                    # Parallel chunk processing
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    summaries_by_idx = {}
                    chunk_times = []
                    
                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                        futures = {}
                        for idx, chunk in enumerate(chunks):
                            if not chunk.strip():
                                continue
                            future = executor.submit(
                                _summarize_chunk_timed, chunk, model_name, level, llm_client
                            )
                            futures[future] = idx
                        
                        completed = 0
                        for future in as_completed(futures):
                            idx = futures[future]
                            chunk_sum, elapsed = future.result()
                            chunk_times.append(elapsed)
                            completed += 1
                            avg_so_far = f" | avg={sum(chunk_times)/len(chunk_times):.1f}s" if chunk_times else ""
                            pbar.set_postfix({"file": filename[:20], "step": f"AI Sum {completed}/{len(futures)}{avg_so_far}"})
                            if chunk_sum:
                                summaries_by_idx[idx] = chunk_sum
                    
                    # Reassemble in original order
                    summaries = [summaries_by_idx[i] for i in sorted(summaries_by_idx.keys())]
                    summary_md = "\n\n".join(summaries)
                    total_t = sum(chunk_times)
                    avg_t = total_t / len(chunk_times) if chunk_times else 0
                    pbar.set_postfix({"file": filename[:20], "step": f"Done", "avg": f"{avg_t:.1f}s/chunk"})
                else:
                    summaries = []
                    chunk_times = []
                    for idx, chunk in enumerate(chunks):
                        if not chunk.strip():
                            continue
                        avg_so_far = f" | avg={sum(chunk_times)/len(chunk_times):.1f}s" if chunk_times else ""
                        pbar.set_postfix({"file": filename[:20], "step": f"AI Sum {idx+1}/{len(chunks)}{avg_so_far}"})
                        t0 = time.time()
                        chunk_sum = summarize_text(chunk, model_name=model_name, level=level, llm_client=llm_client)
                        elapsed = time.time() - t0
                        chunk_times.append(elapsed)
                        if chunk_sum:
                            summaries.append(chunk_sum)
                    summary_md = "\n\n".join(summaries)
                    total_t = sum(chunk_times)
                    avg_t = total_t / len(chunk_times) if chunk_times else 0
                    pbar.set_postfix({"file": filename[:20], "step": f"Done", "avg": f"{avg_t:.1f}s/chunk"})
            
            pbar.set_postfix({"file": filename[:20], "step": "Saving"})
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary_md)
                
        except Exception as e:
            error_details = str(e)
            tqdm.write(f"\n[{filename}] Failed with error: {error_details}")
            log_error(input_path, f"Unexpected exception: {error_details}")
            continue

    # Unload model from VRAM now that processing is complete (Ollama only)
    llm_client.unload_model(model_name)


def _summarize_chunk_timed(chunk, model_name, level, llm_client):
    """Wrapper that returns (summary, elapsed_time) for use in thread pool."""
    t0 = time.time()
    result = summarize_text(chunk, model_name=model_name, level=level, llm_client=llm_client)
    elapsed = time.time() - t0
    return result, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and summarize documents.")
    parser.add_argument("--source", default="source", help="Source directory containing the documents to process (default: source).")
    parser.add_argument("--dest", default="destination", help="Destination directory for the Markdown summaries (default: destination).")
    parser.add_argument("--model", default="gemma3:12b", help="Model to use (default: gemma3:12b).")
    parser.add_argument("--level", type=int, default=9, help="Summarization detail level from 1 to 10. 0 means no summarization (saves raw text). Default: 9.")
    parser.add_argument("--qa", action="store_true", help="Enable QA Dataset Generation mode (reads .md files from source and creates a dataset in dest).")
    parser.add_argument("--full", action="store_true", help="Run the full pipeline: Document summarization followed by QA Dataset generation.")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all files, ignoring resume state.")
    
    # vLLM / provider options
    parser.add_argument("--provider", default="ollama", choices=["ollama", "vllm"], help="LLM provider to use: 'ollama' (local, default) or 'vllm' (remote OpenAI-compatible server).")
    parser.add_argument("--vllm-url", default="http://localhost:8000", help="vLLM server URL (default: http://localhost:8000). Only used when --provider=vllm.")
    parser.add_argument("--vllm-key", default="", help="API key for the vLLM server (optional). Only used when --provider=vllm.")
    parser.add_argument("--threads", type=int, nargs='?', const=5, default=None, help="Enable multithreaded chunk processing. Without a value, defaults to 5 threads. You can specify a custom number (e.g. --threads 8). Omit this flag entirely for sequential processing.")
    
    # OCR engine options (PaddleOCR integration)
    parser.add_argument("--ocr-engine", default="paddleocr", choices=["paddleocr", "tesseract"], help="OCR engine to use: 'paddleocr' (PP-OCRv5, deep learning, default) or 'tesseract' (legacy). PaddleOCR is significantly more accurate.")
    parser.add_argument("--ocr-device", default="cpu", choices=["cpu", "gpu"], help="Device for OCR inference: 'cpu' (default) or 'gpu'. Only affects PaddleOCR.")
    parser.add_argument("--ocr-lang", default="en", help="Language hint for OCR engine (default: en). PaddleOCR supports 109 languages natively. Examples: en, fr, ch, de, es, ja, ko, ar.")
    parser.add_argument("--structured", action="store_true", help="Use PP-StructureV3 for layout-aware PDF parsing (extracts tables, formulas, headings as structured Markdown). Requires PaddleOCR with structure support.")
    
    args = parser.parse_args()
    
    # Build the LLM client
    llm_client = LLMClient(
        provider=args.provider,
        vllm_url=args.vllm_url,
        vllm_api_key=args.vllm_key
    )
    num_threads = args.threads if args.threads else 1
    
    # Configure OCR engine
    configure_ocr(engine=args.ocr_engine, device=args.ocr_device, lang=args.ocr_lang)
    
    print(f"LLM Client: {llm_client}" + (f" | Threads: {num_threads}" if num_threads > 1 else ""))
    
    if args.full:
        from qa_generator import generate_qa_dataset
        print("--- Starting FULL Pipeline ---")
        print("\n[Step 1/2] Document Processing")
        process_documents(args.source, args.dest, args.model, args.level, force=args.force, llm_client=llm_client, num_threads=num_threads, structured=args.structured)
        
        print("\n[Step 2/2] Q&A Dataset Generation")
        qa_dest = "dataresults" if args.dest == "destination" else f"{args.dest}_qa"
        generate_qa_dataset(args.dest, qa_dest, args.model, llm_client=llm_client, num_threads=num_threads)
        print("\n--- Full Pipeline Complete ---")
    elif args.qa:
        from qa_generator import generate_qa_dataset
        print("--- Starting Q&A Dataset Generation ---")
        
        # Override defaults specifically for QA mode if they haven't been manually changed
        qa_source = "destination" if args.source == "source" else args.source
        qa_dest = "dataresults" if args.dest == "destination" else args.dest
        
        generate_qa_dataset(qa_source, qa_dest, args.model, llm_client=llm_client, num_threads=num_threads)
    else:
        print("--- Starting document processing ---")
        process_documents(args.source, args.dest, args.model, args.level, force=args.force, llm_client=llm_client, num_threads=num_threads, structured=args.structured)
