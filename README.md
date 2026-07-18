# Dataminder: Offline Document Ingestion, OCR & RAG Dataset Generator via Local LLMs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/AI-Ollama-black?logo=ollama&logoColor=white)
![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR%20PP--OCRv5-blue?logo=paddle&logoColor=white)
![Tesseract OCR](https://img.shields.io/badge/OCR-Tesseract%20(fallback)-green)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/Web-BeautifulSoup-red)

**Dataminder** is a robust, privacy-first, offline document ingestion pipeline and fine-tuning dataset generator. It automates the extraction of text from diverse file formats (PDFs, Office Documents, Archives, Images, Audio, Video, HTML) using **Tesseract OCR** by default (with **PaddleOCR PP-OCRv5** as a deep-learning option) and speech-to-text transcription via **faster-whisper** (Whisper). Using local Large Language Models (LLMs) via **Ollama** or **vLLM**, Dataminder generates highly detailed, non-redundant Markdown summaries and structured Q&A JSON datasets (Alpaca/ShareGPT format) perfect for RAG (Retrieval-Augmented Generation) architectures and model fine-tuning.

## Why Dataminder?
- **100% Offline & Private:** No API keys, no cloud data leaks. Everything runs locally on your machine.
- **Automated ML Dataset Creation:** Seamlessly converts unstructured local documents into deduplicated Alpaca JSON datasets for AI training.
- **Deep Learning OCR:** PaddleOCR PP-OCRv5 available as an option — 13% more accurate than previous generation, 109 languages, auto orientation/distortion correction.
- **Structured Document Parsing:** Optional PP-StructureV3 mode extracts tables, formulas, and layout as structured Markdown.
- **Smart OCR Fallback:** Automatically detects scanned PDFs/images and applies OCR. Falls back to Tesseract or PaddleOCR depending on active options and availability.
- **Fail-Safe Pipeline:** Continuous processing with automatic skipped files, resume support, and comprehensive error logging.

## Directory Structure

All data lives inside the `data/` directory:

```
dataminder/
├── data/
│   ├── source/          # Input: place your documents here (pdf, docx, epub, images, audio, video...)
│   ├── export/          # Output: AI-generated Markdown summaries (.md)
│   ├── results/         # Output: final QA datasets after deduplication
│   │   ├── dataset_qa.json            # Deduplicated Q&A pairs (Alpaca format)
│   │   ├── dataset_qa.md              # Human-readable Q&A pairs
│   │   ├── dataset_qa_raw.jsonl       # Raw Q&A pairs before deduplication
│   │   └── dataset_qa_enriched.json   # Enriched dataset for LLM fine-tuning
│   └── error.log        # Centralized error log
├── main.py              # Entry point
├── extractor.py         # Text extraction from all formats
├── summarizer.py        # AI summarization via LLM
├── qa_generator.py      # Q&A dataset generation & enrichment
├── llm_client.py        # Unified LLM client (Ollama / vLLM)
├── ocr_engines.py       # OCR abstraction (Tesseract / PaddleOCR)
├── transcriber.py       # Audio/video transcription (faster-whisper)
└── logger.py            # Centralized error logging
```

## Supported Formats

**Documents:** `.txt`, `.md`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.pptx`, `.pdf`, `.epub`, `.cbz`, `.cbr`, `.html`, `.htm`, `.chm`

**Images (OCR):** `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`

**Audio (Transcription):** `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.wma`, `.aac`

**Video (Transcription):** `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`, `.wmv`

## Prerequisites

- Linux (Ubuntu/Debian) or macOS
- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally.
- An Ollama model downloaded (e.g., `gemma3:4b-it-q4_K_M`, `gemma3:12b`, `mistral`). To download a model, run in your terminal: `ollama pull gemma3:4b-it-q4_K_M`.

## Installation

First, clone the repository to your local machine:
```bash
git clone https://github.com/omnia-projetcs/dataminder.git
cd dataminder
```

An automated script is provided to install both the system dependencies (necessary to read scanned PDFs) and the Python dependencies in a virtual environment.

Run in your terminal:
```bash
chmod +x install.sh
./install.sh
```

> **Note:** The installation script uses `sudo` to install `tesseract-ocr` and `poppler-utils`. It will prompt you for your password.

### PaddleOCR Installation (Recommended)

PaddleOCR is included in the Python dependencies (`requirements.txt`). It will be installed automatically with `pip install -r requirements.txt`.

For GPU acceleration (CUDA), install the GPU variant of PaddlePaddle:
```bash
pip install paddlepaddle-gpu
```

## Usage

1. **Activate the virtual environment**:
   Before running the script, make sure to activate the Python virtual environment:
   ```bash
   source venv/bin/activate
   ```

### Quick Examples

Here are the most common commands you will need:

- **Standard Extraction:** Extract text from documents in `data/source/` and save summaries to `data/export/`:
  ```bash
  python main.py
  ```
- **Q&A Generation:** Read the summaries from `data/export/` and create a Q&A dataset in `data/results/`:
  ```bash
  python main.py --qa
  ```
- **The Full Pipeline:** Do both steps above automatically in one go, including post-deduplication enrichment:
  ```bash
  python main.py --full
  ```
- **Custom Folders & Model:** Run extraction with a specific AI model and custom folders:
  ```bash
  python main.py --source ./my_files --dest ./my_summaries --model llama3
  ```
- **Adjust Summarization Level:** Control the summary length from 1 (brief) to 10 (exhaustive). 0 saves the raw text without AI processing:
  ```bash
  python main.py --level 3
  ```
- **Use GPU for OCR:** Accelerate OCR processing with CUDA:
  ```bash
  python main.py --ocr-device gpu
  ```
- **Structured PDF parsing:** Extract tables, formulas, and layout from PDFs:
  ```bash
  python main.py --structured
  ```
- **Use PaddleOCR instead of Tesseract:**
  ```bash
  python main.py --paddleocr
  ```
- **Transcribe audio files:** Process audio files (mp3, wav, ogg, flac...) from `data/source/`:
  ```bash
  python main.py
  ```
- **Transcribe videos:** Process video files (mp4, mkv, avi...) with a specific Whisper model:
  ```bash
  python main.py --whisper-model small
  ```
- **Transcribe in a specific language:**
  ```bash
  python main.py --whisper-lang fr
  ```
- **GPU-accelerated transcription:**
  ```bash
  python main.py --whisper-device cuda --whisper-model medium
  ```
- **Control enrichment ratio** (default 30% of deduplicated pairs):
  ```bash
  python main.py --full --enrich-ratio 0.5
  ```
- **Standalone enrichment** on an existing dataset (with resume support):
  ```bash
  python main.py --enrich
  python main.py --enrich --enrich-ratio 0.5   # Enrich 50%
  ```

### Detailed Commands

2. **Run the processing**:
   The script can be run without any arguments. By default, it looks for documents in `data/source/`, saves the summaries in `data/export/`, uses the `gemma3:4b-it-q4_K_M` model, and applies a summarization level of `9`.
   ```bash
   python main.py
   ```
   
   You can customize this behavior using command line arguments:
   ```bash
   python main.py --source ./my_documents --dest ./my_summaries --model llama3 --level 7
   ```
   
   The output files will keep their original name but with a `.md` extension (e.g., `document.md`). If a summary already exists for a file, it will automatically be skipped, which makes resuming interrupted jobs easy!

## Q&A Dataset Generation

If you want to use the generated summaries to fine-tune an AI model, Dataminder includes a dedicated Q&A generator mode.
It will read all `.md` files in the source directory and use the AI to generate high-quality, non-redundant Question/Answer pairs.

To use this mode, simply add the `--qa` flag:
```bash
python main.py --qa
```

By default in `--qa` mode, Dataminder will automatically read the `.md` files from `data/export/` (where your summaries are) and will output the dataset files into `data/results/`. You can still override these with `--source` and `--dest` if needed.

This will create three files in `data/results/`:
1. `dataset_qa.json` : A structured dataset in standard JSON format (Alpaca style) ready for fine-tuning.
2. `dataset_qa.md` : A human-readable Markdown file containing all the Q&A pairs.
3. `dataset_qa_raw.jsonl` : The raw Q&A pairs before deduplication (used for resume support).

**Note:** The script will first generate all questions for all files, and then run a programmatic deduplication pass (removing any questions that are 85% similar to each other) before saving the final `.json` and `.md` files.

## Full Pipeline (--full)

If you want Dataminder to automatically chain the entire process (extract & summarize all documents, generate the deduplicated Q&A dataset, AND enrich a subset for LLM fine-tuning), you can use the `--full` option:
```bash
python main.py --full
```
This will:
1. Process all documents from `data/source/` and put the `.md` summaries in `data/export/`.
2. Automatically read the `data/export/` summaries and generate the deduplicated dataset in `data/results/`.
3. Enrich a fraction of the dataset (default: 30%) with structured cybersecurity context (MITRE ATT&CK, CVSS, IOC, etc.) and save as `dataset_qa_enriched.json`.

### Post-Deduplication Enrichment

The enrichment phase transforms Q&A pairs into structured reports with:
- **Technical context** (logs, tool output, realistic scenarios)
- **MITRE ATT&CK** technique references (TXXXX)
- **CVSS** scoring
- **Recommendations** and **IOC** (Indicators of Compromise)

You can control the enrichment ratio with `--enrich-ratio`:
```bash
python main.py --full --enrich-ratio 0.5   # Enrich 50% of pairs
python main.py --full --enrich-ratio 0     # Skip enrichment
```

### Standalone Enrichment (--enrich)

If you already have a `dataset_qa.json` in `data/results/` and want to run (or re-run) enrichment without regenerating the QA dataset:
```bash
python main.py --enrich
python main.py --enrich --enrich-ratio 0.5
python main.py --enrich --model mistral     # Use a different model
```

**Resume support:** If the enrichment is interrupted (crash, Ctrl+C), rerunning the same command will automatically resume from where it left off. Progress is saved incrementally to `dataset_qa_enriched.json`.

## Configuration

You can change the AI model directly via the `--model` parameter when launching the script:

```bash
python main.py --model mistral
```

## Remote vLLM Support

Dataminder supports using a remote **vLLM** server (or any OpenAI-compatible API) as an alternative to Ollama. This is useful when you want to leverage a powerful GPU server for inference.

### Basic Usage

```bash
# Use a remote vLLM server
python main.py --provider vllm --vllm-url http://my-server:8000 --model my-model-name

# With an API key
python main.py --provider vllm --vllm-url http://my-server:8000 --vllm-key sk-mykey --model my-model-name
```

### Multithreading (Parallel Chunk Processing)

When using an LLM server that can handle concurrent requests, you can speed up processing by using multiple threads. Each chunk of a document will be sent in parallel. The LLM client reuses thread-local HTTP sessions so model workers avoid reconnecting for every chunk:

```bash
# Process chunks with 4 threads
python main.py --provider vllm --vllm-url http://my-server:8000 --model my-model --threads 4

# Full pipeline with 8 threads
python main.py --full --provider vllm --vllm-url http://my-server:8000 --model my-model --threads 8

# QA generation with multithreading
python main.py --qa --provider vllm --vllm-url http://my-server:8000 --model my-model --threads 4
```

> **Note:** Multithreading is most beneficial with servers that handle concurrent requests (for example, vLLM or an Ollama setup configured for parallel model work). If your local Ollama server serializes requests, `--threads > 1` may not improve throughput.

### All CLI Options

| Argument | Default | Description |
|---|---|---|
| `--source` | `data/source` | Source directory for documents |
| `--dest` | `data/export` | Destination directory for summaries |
| `--model` | `gemma3:4b-it-q4_K_M` | Model name to use |
| `--level` | `9` | Summarization detail (0-10, 0 = raw text) |
| `--provider` | `ollama` | LLM provider: `ollama` or `vllm` |
| `--vllm-url` | `http://localhost:8000` | vLLM server URL |
| `--vllm-key` | *(empty)* | API key for vLLM (optional) |
| `--ollama-url` | `OLLAMA_HOST` or `http://localhost:11434` | Ollama server URL (Ollama provider only) |
| `--threads` | *(off)* | Enable parallel chunk processing (default: 5 threads if activated, e.g. `--threads` or `--threads 8`) |
| `--enrich-ratio` | `0.3` | Fraction of deduplicated pairs to enrich (0 = skip, 1.0 = all). Used with `--full` or `--enrich` |
| `--ocr-engine` | `tesseract` | OCR engine: `tesseract` (legacy, default) or `paddleocr` (PP-OCRv5, deep learning) |
| `--paddleocr` | *(flag)* | Enable PaddleOCR PP-OCRv5 (deep learning) instead of Tesseract |
| `--ocr-device` | `cpu` | Device for OCR inference: `cpu` or `gpu` (PaddleOCR only) |
| `--ocr-lang` | `en` | Language hint for OCR (e.g., `en`, `fr`, `ch`, `de`, `ja`, `ko`) |
| `--structured` | *(flag)* | Use PP-StructureV3 for layout-aware PDF parsing |
| `--whisper-model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--whisper-device` | `cpu` | Device for Whisper inference: `cpu` or `cuda` |
| `--whisper-lang` | *(auto)* | Source language for transcription (e.g., `en`, `fr`). Default: auto-detect |
| `--qa` | *(flag)* | Enable QA dataset generation mode |
| `--full` | *(flag)* | Run full pipeline (summarize + QA + enrichment) |
| `--enrich` | *(flag)* | Run enrichment only on existing `dataset_qa.json` (with resume support) |
| `--force` | *(flag)* | Force reprocessing of all files |

## OCR Engine Details

### Tesseract OCR (Default)

Tesseract is the default OCR engine. It uses a lightweight, traditional OCR framework which has wide language support and low resource usage.

### PaddleOCR PP-OCRv5 (Deep Learning Option)

PaddleOCR can be enabled using `--paddleocr`. It uses deep learning models from [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for significantly better accuracy:

- **PP-OCRv5 text detection** — Deep learning based text zone detection
- **PP-OCRv5 text recognition** — 13% more accurate than PP-OCRv4, 109 languages natively
- **Auto orientation correction** — Detects and corrects rotated documents
- **Auto distortion correction** — Fixes warped/photographed documents
- **Direct PDF processing** — No intermediate image conversion needed
- **GPU acceleration** — Use `--ocr-device gpu` for faster processing

### PP-StructureV3 (Structured Parsing)

When `--structured` is enabled, Dataminder uses PP-StructureV3 for layout-aware document parsing:

- Extracts headings, paragraphs, tables, images, and formulas
- Converts tables to Markdown tables
- Converts formulas to LaTeX
- Much better output quality for complex documents with mixed content

### PaddleOCR Fallback

PaddleOCR automatically falls back to Tesseract if it fails or if the deep learning packages are not installed/available.

## Audio & Video Transcription (faster-whisper)

Dataminder uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text transcription. It is a CTranslate2-based reimplementation of OpenAI's Whisper, offering ~4x faster inference with ~3x less memory while maintaining identical accuracy.

### How It Works

1. **Audio files** (mp3, wav, ogg, flac, m4a, wma, aac) are transcribed directly
2. **Video files** (mp4, mkv, avi, mov, webm, flv, wmv) have their audio track extracted via `ffmpeg`, then transcribed
3. The transcribed text is then processed through the same summarization pipeline as any other document

### Model Selection

| Model | Size | Speed | Accuracy | Best For |
|---|---|---|---|---|
| `tiny` | ~40 MB | ⚡⚡⚡⚡ | ★★ | Quick previews, low resources |
| `base` | ~140 MB | ⚡⚡⚡ | ★★★ | **Default — good balance** |
| `small` | ~460 MB | ⚡⚡ | ★★★★ | Better accuracy, still fast |
| `medium` | ~1.5 GB | ⚡ | ★★★★★ | High accuracy |
| `large-v3` | ~3 GB | 🐢 | ★★★★★★ | Maximum accuracy |

### Requirements

- **ffmpeg**: Required for video transcription (audio extraction). Install with `sudo apt-get install ffmpeg` or `brew install ffmpeg`.
- **faster-whisper**: Included in `requirements.txt`. Installed automatically.
- **GPU (optional)**: Use `--whisper-device cuda` for GPU-accelerated transcription. Requires CUDA-compatible GPU.