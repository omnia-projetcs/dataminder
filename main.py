import os
# Disable oneDNN (MKLDNN) to prevent NotImplementedError in PaddlePaddle 3.3.0+ CPU inference
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from document_ir import build_chunks, chunks_to_jsonl, sha256_file
from document_parser import extract_document
from extractor import (
    SUPPORTED_EXTENSIONS,
    configure_ocr,
    configure_whisper,
    extraction_config,
)
from summarizer import SUMMARY_PROMPT_REVISION, summarize_text
from llm_client import LLMClient
from logger import log_error as _log_error
from processing_manifest import (
    ProcessingManifest,
    atomic_write_text,
    output_paths,
    relative_source_id,
    stable_fingerprint,
)
from run_report import DEFAULT_RUN_REPORT, PipelineRunReport
from tqdm import tqdm

def log_error(filepath, error_msg):
    _log_error(filepath, error_msg, category="PIPELINE")

def process_documents(
    source_dir,
    dest_dir,
    model_name,
    level=7,
    force=False,
    llm_client=None,
    num_threads=1,
    structured=False,
    parser_backend="native",
    marker_mode="fast",
    report_path=None,
):
    if llm_client is None:
        llm_client = LLMClient(provider="ollama")

    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist. Creating it now...")
        os.makedirs(source_dir, exist_ok=True)

    os.makedirs(dest_dir, exist_ok=True)
    resolved_report_path = report_path or os.path.join(dest_dir, DEFAULT_RUN_REPORT)
    run_report = PipelineRunReport(source_dir, dest_dir)
    pipeline_config = {
        "pipeline_revision": 4,
        "summary_prompt_revision": SUMMARY_PROMPT_REVISION,
        "model": model_name,
        "summary_level": level,
        "structured": structured,
        "parser_backend": parser_backend,
        "marker_mode": marker_mode,
        "extractor": extraction_config(),
        "llm_generation": (
            llm_client.generation_config()
            if hasattr(llm_client, "generation_config")
            else {
                "provider": getattr(llm_client, "provider", "unknown"),
            }
        ),
    }
    run_report.set_configuration(pipeline_config)
    pipeline_fingerprint = stable_fingerprint(pipeline_config)
    print(f"Scanning '{source_dir}' recursively...")
    
    files_to_process = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files_to_process.append(os.path.join(root, file))
    files_to_process.sort()
    
    if not files_to_process:
        print(f"No supported documents found in '{source_dir}'.")
        listed = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        print(f"Supported: {listed}")
        return run_report.write(resolved_report_path)
        
    manifest = ProcessingManifest(dest_dir)
    # Resume only when content, configuration, and every expected output match.
    skipped = 0
    work_items = []
    for filepath in files_to_process:
        source_id = relative_source_id(filepath, source_dir)
        outputs = output_paths(filepath, source_dir, dest_dir)
        try:
            source_sha256 = sha256_file(filepath)
        except OSError as exc:
            log_error(filepath, f"Could not hash source file: {exc}")
            run_report.add_document(
                source_id=source_id,
                status="failed",
                error=f"Could not hash source file: {exc}",
            )
            continue
        if (
            not force
            and manifest.is_current(
                source_id,
                source_sha256=source_sha256,
                pipeline_fingerprint=pipeline_fingerprint,
                outputs=outputs,
            )
        ):
            skipped += 1
            run_report.add_document(
                source_id=source_id,
                status="skipped",
                source_sha256=source_sha256,
                outputs={
                    name: os.path.relpath(path, dest_dir).replace(os.sep, "/")
                    for name, path in outputs.items()
                },
            )
            continue
        work_items.append((filepath, source_id, source_sha256, outputs))
    
    if skipped:
        print(f"Resuming: {skipped} unchanged files already processed (use --force to reprocess). {len(work_items)} remaining.")
    
    if not work_items:
        print("All files already processed. Nothing to do.")
        return run_report.write(resolved_report_path)
    
    print(f"Processing {len(work_items)} documents...")
    if num_threads > 1:
        print(f"Using {num_threads} threads for parallel chunk processing.")
    
    pbar = tqdm(work_items, desc="Processing", unit="doc")
    for input_path, source_id, source_sha256, outputs in pbar:
        filename = os.path.basename(input_path)
        document_started = time.perf_counter()
        
        try:
            pbar.set_postfix({"file": filename[:20], "step": "Extracting text"})
            extraction_started = time.perf_counter()
            document = extract_document(
                input_path,
                parser=parser_backend,
                marker_mode=marker_mode,
                structured=structured,
                source_sha256=source_sha256,
            )
            extraction_elapsed = time.perf_counter() - extraction_started
            document.metadata["source_relative_path"] = source_id
            document.metadata["pipeline_config"] = pipeline_config
            text = document.text
            
            if not text.strip():
                tqdm.write(f"\n[{filename}] No text could be extracted. Skipping.")
                log_error(input_path, "No text extracted (unsupported or OCR failed).")
                run_report.add_document(
                    source_id=source_id,
                    status="failed",
                    source_sha256=source_sha256,
                    parser_requested=parser_backend,
                    parser_used=document.metadata.get("parser", parser_backend),
                    error="No text extracted (unsupported or OCR failed).",
                    extraction_seconds=round(extraction_elapsed, 6),
                    elapsed_seconds=round(
                        time.perf_counter() - document_started, 6
                    ),
                )
                continue

            rag_chunks = build_chunks(document)
            summary_started = time.perf_counter()
            if level == 0:
                pbar.set_postfix({"file": filename[:20], "step": "Saving Raw Text"})
                summary_md = text
            else:
                # Use document-aware chunks so tables/code/page provenance survive.
                chunks = [chunk["text"] for chunk in rag_chunks]

                if len(chunks) <= 1:
                    pbar.set_postfix({"file": filename[:20], "step": f"AI Summarizing (L{level})"})
                    t0 = time.time()
                    summary_md = summarize_text(text, model_name=model_name, level=level, llm_client=llm_client)
                    elapsed = time.time() - t0
                    pbar.set_postfix({"file": filename[:20], "step": "Done", "time": f"{elapsed:.1f}s"})
                elif num_threads > 1:
                    # Parallel chunk processing
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
                    pbar.set_postfix({"file": filename[:20], "step": "Done", "avg": f"{avg_t:.1f}s/chunk"})
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
                    pbar.set_postfix({"file": filename[:20], "step": "Done", "avg": f"{avg_t:.1f}s/chunk"})
            summary_elapsed = time.perf_counter() - summary_started
            
            pbar.set_postfix({"file": filename[:20], "step": "Saving"})
            atomic_write_text(outputs["markdown"], summary_md)
            atomic_write_text(outputs["chunks"], chunks_to_jsonl(rag_chunks))
            atomic_write_text(
                outputs["document"],
                json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n",
            )
            manifest.mark_success(
                source_id,
                source_sha256=source_sha256,
                pipeline_fingerprint=pipeline_fingerprint,
                outputs=outputs,
                document_id=document.id,
            )
            run_report.add_document(
                source_id=source_id,
                status="success",
                source_sha256=source_sha256,
                document_id=document.id,
                parser_requested=parser_backend,
                parser_used=document.metadata.get("parser", parser_backend),
                extraction_methods=document.metadata.get(
                    "extraction_methods", []
                ),
                char_count=len(text),
                block_count=len(document.blocks),
                chunk_count=len(rag_chunks),
                diagnostic_count=len(document.diagnostics),
                extraction_seconds=round(extraction_elapsed, 6),
                summary_seconds=round(summary_elapsed, 6),
                elapsed_seconds=round(
                    time.perf_counter() - document_started, 6
                ),
                outputs={
                    name: os.path.relpath(path, dest_dir).replace(os.sep, "/")
                    for name, path in outputs.items()
                },
            )

        except Exception as e:
            error_details = str(e)
            tqdm.write(f"\n[{filename}] Failed with error: {error_details}")
            log_error(input_path, f"Unexpected exception: {error_details}")
            run_report.add_document(
                source_id=source_id,
                status="failed",
                source_sha256=source_sha256,
                parser_requested=parser_backend,
                error=error_details,
                elapsed_seconds=round(
                    time.perf_counter() - document_started, 6
                ),
            )
            continue

    # Unload model from VRAM now that processing is complete (Ollama only)
    llm_client.unload_model(model_name)
    return run_report.write(resolved_report_path)


def _summarize_chunk_timed(chunk, model_name, level, llm_client):
    """Wrapper that returns (summary, elapsed_time) for use in thread pool."""
    t0 = time.time()
    result = summarize_text(chunk, model_name=model_name, level=level, llm_client=llm_client)
    elapsed = time.time() - t0
    return result, elapsed


def build_model_data_products(
    *,
    cyber_source,
    finance_source,
    rag_dir,
    training_dir,
    colab_dir,
    domain="both",
    chunk_max_chars=3200,
    chunk_overlap_chars=320,
    min_quality=0.80,
    min_training_chars=400,
    min_training_words=50,
    max_chunks_per_document=600,
):
    """Build conservative RAG and model-enrichment products from originals.

    This path is deliberately independent from the summarization/Q&A pipeline:
    generated summaries are never used as factual training sources.
    """
    from scripts.build_rag_databases import build_database
    from scripts.export_colab_messages import convert_file
    from scripts.export_model_enrichment import export_database

    if domain not in {"both", "cyber", "finance"}:
        raise ValueError("domain must be one of: both, cyber, finance")
    if chunk_max_chars < 200:
        raise ValueError("chunk_max_chars must be at least 200")
    if not 0 <= chunk_overlap_chars < chunk_max_chars:
        raise ValueError(
            "chunk_overlap_chars must be non-negative and smaller than "
            "chunk_max_chars"
        )
    if not 0 <= min_quality <= 1:
        raise ValueError("min_quality must be between 0 and 1")
    if min_training_chars < 400:
        raise ValueError(
            "min_training_chars must be at least 400 for grounded Colab "
            "continuations"
        )
    if min_training_words < 30:
        raise ValueError("min_training_words must be at least 30")
    if max_chunks_per_document < 1:
        raise ValueError("max_chunks_per_document must be positive")

    source_paths = {
        "cyber": Path(cyber_source),
        "finance": Path(finance_source),
    }
    rag_dir = Path(rag_dir)
    training_dir = Path(training_dir)
    colab_dir = Path(colab_dir)
    selected_domains = (
        ("cyber", "finance") if domain == "both" else (domain,)
    )

    report = {
        "schema_version": 1,
        "pipeline": "source-grounded-model-data",
        "policies": {
            "source": "original_documents_only",
            "generated_summaries": "excluded",
            "generated_qa": "excluded",
            "ocr": "native_text_only",
            "document_deduplication": "sha256",
            "chunk_deduplication": "canonical_sha256",
            "rights_status": "unknown_review_required",
        },
        "domains": {},
    }

    for selected_domain in selected_domains:
        source_root = source_paths[selected_domain]
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"{selected_domain} source directory not found: {source_root}"
            )

        rag_path = rag_dir / f"{selected_domain}_rag.sqlite"
        training_path = (
            training_dir / f"{selected_domain}_model_enrichment.jsonl"
        )
        colab_path = (
            colab_dir / f"{selected_domain}_colab_messages.jsonl"
        )
        print(
            f"\n[{selected_domain}] Building RAG database from originals: "
            f"{source_root}"
        )
        rag_report = build_database(
            domain=selected_domain,
            source_root=source_root,
            output_path=rag_path,
            max_chars=chunk_max_chars,
            overlap_chars=chunk_overlap_chars,
        )

        print(f"[{selected_domain}] Exporting filtered model corpus")
        training_report = export_database(
            database=rag_path,
            output=training_path,
            expected_domain=selected_domain,
            min_quality=min_quality,
            min_chars=min_training_chars,
            min_words=min_training_words,
            max_chunks_per_document=max_chunks_per_document,
        )

        print(f"[{selected_domain}] Exporting grounded Colab conversations")
        colab_report = convert_file(
            source=training_path,
            output=colab_path,
            expected_domain=selected_domain,
        )
        report["domains"][selected_domain] = {
            "rag": rag_report,
            "training": training_report,
            "colab": colab_report,
        }

    return report


def default_finance_source(project_root):
    """Prefer the external original corpus, else the in-repo finance markdown."""
    root = Path(project_root)
    external = root.parent / "DATASETS" / "FINANCE" / "FINANCE_DOCS"
    local = root / "data" / "datas-finance"
    if external.is_dir() or not local.is_dir():
        return str(external)
    return str(local)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Extract and summarize documents.")
    parser.add_argument("--source", default=os.path.join("data", "source"), help="Source directory containing the documents to process (default: data/source).")
    parser.add_argument("--dest", default=os.path.join("data", "export"), help="Destination directory for the Markdown summaries (default: data/export).")
    parser.add_argument("--model", default="gemma3:4b-it-q4_K_M", help="Model to use (default: gemma3:4b-it-q4_K_M).")
    parser.add_argument("--level", type=int, default=9, help="Summarization detail level from 1 to 10. 0 means no summarization (saves raw text). Default: 9.")
    parser.add_argument("--qa", action="store_true", help="Enable QA Dataset Generation mode (reads RAG chunks when available, otherwise Markdown summaries).")
    parser.add_argument("--full", action="store_true", help="Run the full pipeline: Document summarization followed by QA Dataset generation.")
    parser.add_argument("--enrich", action="store_true", help="Run enrichment only on an existing dataset_qa.json in data/results/.")
    parser.add_argument(
        "--build-model-data",
        action="store_true",
        help=(
            "Build source-grounded RAG, documentary training, and Colab "
            "datasets without generated summaries or Q&A."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all files, ignoring resume state.")

    # Conservative model-data pipeline
    parser.add_argument(
        "--model-data-domain",
        default="both",
        choices=["both", "cyber", "finance"],
        help="Domain(s) to build with --build-model-data (default: both).",
    )
    parser.add_argument(
        "--cyber-source",
        default=str(project_root / "data" / "source" / "00_DATA_SOURCE_LLM"),
        help="Original Cyber source directory.",
    )
    parser.add_argument(
        "--finance-source",
        default=default_finance_source(project_root),
        help=(
            "Original Finance source directory. Defaults to "
            "../DATASETS/FINANCE/FINANCE_DOCS when present, otherwise "
            "data/datas-finance."
        ),
    )
    parser.add_argument(
        "--rag-dir",
        default=str(project_root / "data" / "rag"),
        help="Directory for the two SQLite RAG databases.",
    )
    parser.add_argument(
        "--training-dir",
        default=str(project_root / "data" / "model_enrichment"),
        help="Directory for filtered documentary training JSONL files.",
    )
    parser.add_argument(
        "--colab-dir",
        default=str(project_root / "data" / "model_enrichment_colab"),
        help="Directory for Colab-compatible messages JSONL files.",
    )
    parser.add_argument(
        "--rag-max-chars",
        type=int,
        default=3200,
        help="Maximum RAG chunk size in characters (default: 3200).",
    )
    parser.add_argument(
        "--rag-overlap-chars",
        type=int,
        default=320,
        help="RAG chunk overlap in characters (default: 320).",
    )
    parser.add_argument(
        "--training-min-quality",
        type=float,
        default=0.80,
        help="Minimum structural quality score for training (default: 0.80).",
    )
    parser.add_argument(
        "--training-min-chars",
        type=int,
        default=400,
        help="Minimum training example size in characters (default: 400).",
    )
    parser.add_argument(
        "--training-min-words",
        type=int,
        default=50,
        help="Minimum training example word count (default: 50).",
    )
    parser.add_argument(
        "--max-chunks-per-document",
        type=int,
        default=600,
        help=(
            "Maximum examples per document and normalized title to limit "
            "single-source dominance (default: 600)."
        ),
    )

    # Document parser options
    parser.add_argument(
        "--parser",
        dest="parser_backend",
        default="native",
        choices=["native", "marker", "auto"],
        help="Document parser: native (default), marker (strict), or auto (Marker with native fallback).",
    )
    parser.add_argument(
        "--marker-mode",
        default="fast",
        choices=["fast", "balanced"],
        help="Marker conversion mode when --parser=marker/auto (default: fast).",
    )
    
    # vLLM / provider options
    parser.add_argument("--provider", default="ollama", choices=["ollama", "vllm"], help="LLM provider to use: 'ollama' (local, default) or 'vllm' (remote OpenAI-compatible server).")
    parser.add_argument("--vllm-url", default="http://localhost:8000", help="vLLM server URL (default: http://localhost:8000). Only used when --provider=vllm.")
    parser.add_argument("--vllm-key", default="", help="API key for the vLLM server (optional). Only used when --provider=vllm.")
    parser.add_argument("--ollama-url", default=None, help="Ollama server URL (default: OLLAMA_HOST or http://localhost:11434). Only used when --provider=ollama.")
    parser.add_argument('--threads', type=int, nargs='?', const=5, default=None, help="Enable multithreaded chunk processing. Without a value, defaults to 5 threads. You can specify a custom number (e.g. --threads 8). Omit this flag entirely for sequential processing.")
    parser.add_argument('--timeout', type=int, default=300, help="Timeout in seconds for each LLM call (default: 300). The call will be retried up to 3 times with exponential backoff before giving up.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature from 0 to 2 (default: 0 for reproducibility).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional generation seed passed to Ollama/vLLM.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help=(
            "Path for the document-processing run report "
            "(default: DEST/.dataminder-last-run.json)."
        ),
    )
    
    # OCR engine options (PaddleOCR integration)
    parser.add_argument("--ocr-engine", default="tesseract", choices=["paddleocr", "tesseract"], help="OCR engine to use: 'tesseract' (legacy, default) or 'paddleocr' (PP-OCRv5, deep learning).")
    parser.add_argument("--paddleocr", action="store_true", help="Enable PaddleOCR PP-OCRv5 (deep learning, more accurate) instead of Tesseract.")
    parser.add_argument("--ocr-device", default="cpu", choices=["cpu", "gpu"], help="Device for OCR inference: 'cpu' (default) or 'gpu'. Only affects PaddleOCR.")
    parser.add_argument("--ocr-lang", default="en", help="Language hint for OCR engine (default: en). PaddleOCR supports 109 languages natively. Examples: en, fr, ch, de, es, ja, ko, ar.")
    parser.add_argument("--structured", action="store_true", help="Use PP-StructureV3 for layout-aware PDF parsing (extracts tables, formulas, headings as structured Markdown). Requires PaddleOCR with structure support.")
    parser.add_argument("--ocr-dpi", type=int, default=200, help="DPI resolution for rendering scanned PDF pages during OCR (default: 200). Lower values use less RAM but may reduce accuracy. Range: 72-600.")
    parser.add_argument("--ocr-max-pages", type=int, default=0, help="Maximum number of pages to OCR per PDF (default: 0 = unlimited). Use this to prevent OOM on very large scanned PDFs.")
    
    # Whisper transcription options (audio/video)
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large-v3"], help="Whisper model size for audio/video transcription (default: base). Larger models are more accurate but slower.")
    parser.add_argument("--whisper-device", default="cpu", choices=["cpu", "cuda"], help="Device for Whisper inference: 'cpu' (default) or 'cuda' (GPU).")
    parser.add_argument("--whisper-lang", default=None, help="Source language for transcription (e.g., 'en', 'fr'). Default: auto-detect.")

    # RAG options
    parser.add_argument('--enrich-ratio', type=float, default=0.3, help="Post-deduplication enrichment ratio (default: 0.3)")
    parser.add_argument(
        "--qa-source",
        default="auto",
        choices=["auto", "chunks", "summaries"],
        help="Q&A source: prefer chunks automatically (default), require chunks, or use summaries.",
    )
    parser.add_argument(
        "--enrich-domain",
        default="cyber",
        choices=["cyber", "finance", "generic"],
        help="Domain prompt for --enrich / --full (default: cyber).",
    )
    
    args = parser.parse_args()
    if not 0 <= args.level <= 10:
        parser.error("--level must be between 0 and 10")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")
    if not 0.0 <= args.enrich_ratio <= 1.0:
        parser.error("--enrich-ratio must be between 0 and 1")
    if not 72 <= args.ocr_dpi <= 600:
        parser.error("--ocr-dpi must be between 72 and 600")

    selected_modes = sum(
        bool(mode)
        for mode in (
            args.full,
            args.qa,
            args.enrich,
            args.build_model_data,
        )
    )
    if selected_modes > 1:
        parser.error(
            "--full, --qa, --enrich, and --build-model-data are mutually "
            "exclusive"
        )

    if args.build_model_data:
        print("--- Starting source-grounded model-data pipeline ---")
        model_data_report = build_model_data_products(
            cyber_source=args.cyber_source,
            finance_source=args.finance_source,
            rag_dir=args.rag_dir,
            training_dir=args.training_dir,
            colab_dir=args.colab_dir,
            domain=args.model_data_domain,
            chunk_max_chars=args.rag_max_chars,
            chunk_overlap_chars=args.rag_overlap_chars,
            min_quality=args.training_min_quality,
            min_training_chars=args.training_min_chars,
            min_training_words=args.training_min_words,
            max_chunks_per_document=args.max_chunks_per_document,
        )
        print(json.dumps(model_data_report, ensure_ascii=False, indent=2))
        print("\n--- Source-grounded model-data pipeline complete ---")
        raise SystemExit(0)

    if args.full or args.qa or args.enrich:
        print(
            "WARNING: this legacy mode creates or transforms LLM-generated "
            "Q&A. Audit its output before training. Use --build-model-data "
            "for the source-grounded deterministic pipeline."
        )
    
    # Build the LLM client
    llm_client = LLMClient(
        provider=args.provider,
        vllm_url=args.vllm_url,
        vllm_api_key=args.vllm_key,
        timeout=args.timeout,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        seed=args.seed,
    )
    num_threads = args.threads if args.threads else 1
    
    # Configure OCR engine
    ocr_engine = "paddleocr" if args.paddleocr else args.ocr_engine
    configure_ocr(engine=ocr_engine, device=args.ocr_device, lang=args.ocr_lang, dpi=args.ocr_dpi, max_pages=args.ocr_max_pages)
    
    # Configure Whisper transcription engine
    configure_whisper(model=args.whisper_model, device=args.whisper_device, lang=args.whisper_lang)
    
    print(f"LLM Client: {llm_client}" + (f" | Threads: {num_threads}" if num_threads > 1 else ""))
    
    if args.full:
        from qa_generator import generate_qa_dataset
        print("--- Starting FULL Pipeline ---")
        print("\n[Step 1/2] Document Processing")
        process_documents(
            args.source,
            args.dest,
            args.model,
            args.level,
            force=args.force,
            llm_client=llm_client,
            num_threads=num_threads,
            structured=args.structured,
            parser_backend=args.parser_backend,
            marker_mode=args.marker_mode,
            report_path=args.report,
        )
        
        print("\n[Step 2/2] Q&A Dataset Generation")
        qa_dest = os.path.join("data", "results")
        generate_qa_dataset(
            args.dest,
            qa_dest,
            args.model,
            llm_client=llm_client,
            num_threads=num_threads,
            force=args.force,
            enrich=True,
            enrich_ratio=args.enrich_ratio,
            input_format=args.qa_source,
            enrich_domain=args.enrich_domain,
        )
        print("\n--- Full Pipeline Complete ---")
    elif args.qa:
        from qa_generator import generate_qa_dataset
        print("--- Starting Q&A Dataset Generation ---")
        
        # In QA mode: source defaults to data/export (summaries), dest to data/results
        default_source = os.path.join("data", "source")
        default_dest = os.path.join("data", "export")
        qa_source = default_dest if args.source == default_source else args.source
        qa_dest = os.path.join("data", "results") if args.dest == default_dest else args.dest
        
        generate_qa_dataset(
            qa_source,
            qa_dest,
            args.model,
            llm_client=llm_client,
            num_threads=num_threads,
            force=args.force,
            enrich_ratio=args.enrich_ratio,
            input_format=args.qa_source,
            enrich_domain=args.enrich_domain,
        )
    elif args.enrich:
        from qa_generator import _enrich_after_dedup
        print("--- Starting Standalone Enrichment ---")
        
        results_dir = os.path.join("data", "results")
        dataset_path = os.path.join(results_dir, "dataset_qa.json")
        enriched_path = os.path.join(results_dir, "dataset_qa_enriched.json")
        
        if not os.path.exists(dataset_path):
            print(f"ERROR: {dataset_path} not found. Run --qa or --full first to generate the dataset.")
        else:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                qa_list = json.load(f)
            print(f"Loaded {len(qa_list)} entries from {dataset_path}")
            
            enriched = _enrich_after_dedup(
                qa_list,
                ratio=args.enrich_ratio,
                model_name=args.model,
                llm_client=llm_client,
                output_path=enriched_path,
                domain=args.enrich_domain,
            )
            atomic_write_text(
                enriched_path,
                json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
            )
            enriched_count = sum(1 for e in enriched if isinstance(e.get("_meta"), dict) and e["_meta"].get("enriched"))
            print(f"Saved enriched dataset ({enriched_count}/{len(enriched)} entries) to: {enriched_path}")
            from qa_generator import clean_dataset, prepare_hf_dataset
            cleaned_enriched = clean_dataset(enriched_path)
            if cleaned_enriched:
                prepare_hf_dataset(cleaned_enriched)
            llm_client.unload_model(args.model)
        print("\n--- Enrichment Complete ---")
    else:
        print("--- Starting document processing ---")
        process_documents(
            args.source,
            args.dest,
            args.model,
            args.level,
            force=args.force,
            llm_client=llm_client,
            num_threads=num_threads,
            structured=args.structured,
            parser_backend=args.parser_backend,
            marker_mode=args.marker_mode,
            report_path=args.report,
        )
