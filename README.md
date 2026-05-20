# Dataminder: Offline Document Ingestion, OCR & RAG Dataset Generator via Local LLMs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/AI-Ollama-black?logo=ollama&logoColor=white)
![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR%20PP--OCRv5-blue?logo=paddle&logoColor=white)
![Tesseract OCR](https://img.shields.io/badge/OCR-Tesseract%20(fallback)-green)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/Web-BeautifulSoup-red)

**Dataminder** is a robust, privacy-first, offline document ingestion pipeline and fine-tuning dataset generator. It automates the extraction of text from diverse file formats (PDFs, Office Documents, Archives, Images, HTML) using deep-learning OCR powered by **PaddleOCR PP-OCRv5** (with Tesseract as legacy fallback). Using local Large Language Models (LLMs) via **Ollama** or **vLLM**, Dataminder generates highly detailed, non-redundant Markdown summaries and structured Q&A JSON datasets (Alpaca/ShareGPT format) perfect for RAG (Retrieval-Augmented Generation) architectures and model fine-tuning.

## Why Dataminder?
- **100% Offline & Private:** No API keys, no cloud data leaks. Everything runs locally on your machine.
- **Automated ML Dataset Creation:** Seamlessly converts unstructured local documents into deduplicated Alpaca JSON datasets for AI training.
- **Deep Learning OCR:** Powered by PaddleOCR PP-OCRv5 — 13% more accurate than previous generation, 109 languages, auto orientation/distortion correction.
- **Structured Document Parsing:** Optional PP-StructureV3 mode extracts tables, formulas, and layout as structured Markdown.
- **Smart OCR Fallback:** Automatically detects scanned PDFs/images and applies OCR. Falls back to Tesseract if PaddleOCR is unavailable.
- **Fail-Safe Pipeline:** Continuous processing with automatic skipped files and comprehensive `error.log` generation.

## Supported Formats
`.txt`, `.md`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.pptx`, `.pdf`, `.epub`, `.cbz`, `.cbr`, `.html`, `.htm`, `.chm`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`

## Prerequisites

- Linux (Ubuntu/Debian) or macOS
- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally.
- An Ollama model downloaded (e.g., `gemma3:12b`, `mistral`). To download a model, run in your terminal: `ollama pull gemma3:12b`.

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

- **Standard Extraction:** Extract text from documents in `./source` and save summaries to `./destination`:
  ```bash
  python main.py
  ```
- **Q&A Generation:** Read the summaries from `./destination` and create a Q&A dataset in `./dataresults`:
  ```bash
  python main.py --qa
  ```
- **The Full Pipeline:** Do both steps above automatically in one go:
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
- **Use Tesseract instead of PaddleOCR:**
  ```bash
  python main.py --ocr-engine tesseract
  ```

### Detailed Commands

2. **Run the processing**:
   The script can be run without any arguments. By default, it looks for documents in the `source` directory, saves the summaries in the `destination` directory, uses the `gemma3:12b` model, and applies a summarization level of `7`.
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

By default in `--qa` mode, Dataminder will automatically read the `.md` files from the `destination` folder (where your summaries are) and will output the dataset files into a new `dataresults` folder. You can still override these with `--source` and `--dest` if needed.

This will create two files in the `dataresults` folder:
1. `dataset_qa.json` : A structured dataset in standard JSON format (Alpaca style) ready for fine-tuning.
2. `dataset_qa.md` : A human-readable Markdown file containing all the Q&A pairs.

**Note:** The script will first generate all questions for all files, and then run a programmatic deduplication pass (removing any questions that are 85% similar to each other) before saving the final `.json` and `.md` files.

## Full Pipeline (--full)

If you want Dataminder to automatically chain the two processes (extract & summarize all documents, AND immediately generate the non-redundant Q&A dataset out of them), you can use the `--full` option:
```bash
python main.py --full
```
This will:
1. Process all documents from `source` and put the `.md` summaries in `destination`.
2. Automatically read the `destination` summaries and generate the dataset in `dataresults`.

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

When using a remote vLLM server that can handle concurrent requests, you can speed up processing by using multiple threads. Each chunk of a document will be sent in parallel:

```bash
# Process chunks with 4 threads
python main.py --provider vllm --vllm-url http://my-server:8000 --model my-model --threads 4

# Full pipeline with 8 threads
python main.py --full --provider vllm --vllm-url http://my-server:8000 --model my-model --threads 8

# QA generation with multithreading
python main.py --qa --provider vllm --vllm-url http://my-server:8000 --model my-model --threads 4
```

> **Note:** Multithreading is mainly beneficial with remote servers that handle concurrent requests. When using Ollama locally, requests are serialized by the server, so `--threads > 1` won't improve speed.

### All CLI Options

| Argument | Default | Description |
|---|---|---|
| `--source` | `source` | Source directory for documents |
| `--dest` | `destination` | Destination directory for summaries |
| `--model` | `gemma3:12b` | Model name to use |
| `--level` | `9` | Summarization detail (0-10, 0 = raw text) |
| `--provider` | `ollama` | LLM provider: `ollama` or `vllm` |
| `--vllm-url` | `http://localhost:8000` | vLLM server URL |
| `--vllm-key` | *(empty)* | API key for vLLM (optional) |
| `--threads` | *(off)* | Enable parallel chunk processing (default: 5 threads if activated, e.g. `--threads` or `--threads 8`) |
| `--ocr-engine` | `paddleocr` | OCR engine: `paddleocr` (PP-OCRv5, deep learning) or `tesseract` (legacy) |
| `--ocr-device` | `cpu` | Device for OCR inference: `cpu` or `gpu` (PaddleOCR only) |
| `--ocr-lang` | `en` | Language hint for OCR (e.g., `en`, `fr`, `ch`, `de`, `ja`, `ko`) |
| `--structured` | *(flag)* | Use PP-StructureV3 for layout-aware PDF parsing |
| `--qa` | *(flag)* | Enable QA dataset generation mode |
| `--full` | *(flag)* | Run full pipeline (summarize + QA) |
| `--force` | *(flag)* | Force reprocessing of all files |

## OCR Engine Details

### PaddleOCR PP-OCRv5 (Default)

PaddleOCR is the default OCR engine. It uses deep learning models from [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for significantly better accuracy than Tesseract:

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

### Tesseract (Legacy Fallback)

Tesseract is kept as a fallback engine (`--ocr-engine tesseract`). PaddleOCR automatically falls back to Tesseract if the `paddleocr` package is not installed.