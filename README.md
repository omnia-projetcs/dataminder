# Dataminder: Offline Document Ingestion, OCR & RAG Dataset Generator via Local LLMs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/AI-Ollama-black?logo=ollama&logoColor=white)
![Tesseract OCR](https://img.shields.io/badge/OCR-Tesseract-green)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/Web-BeautifulSoup-red)

**Dataminder** is a robust, privacy-first, offline document ingestion pipeline and fine-tuning dataset generator. It automates the extraction of text from diverse file formats (PDFs, Office Documents, Archives, HTML) including scanned images via Tesseract OCR. Using local Large Language Models (LLMs) via **Ollama**, Dataminder generates highly detailed, non-redundant Markdown summaries and structured Q&A JSON datasets (Alpaca/ShareGPT format) perfect for RAG (Retrieval-Augmented Generation) architectures and model fine-tuning.

## Why Dataminder?
- **100% Offline & Private:** No API keys, no cloud data leaks. Everything runs locally on your machine.
- **Automated ML Dataset Creation:** Seamlessly converts unstructured local documents into deduplicated Alpaca JSON datasets for AI training.
- **Smart OCR Fallback:** Automatically detects scanned PDFs/images and switches to Tesseract OCR.
- **Fail-Safe Pipeline:** Continuous processing with automatic skipped files and comprehensive `error.log` generation.

## Supported Formats
`.txt`, `.md`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.pptx`, `.pdf`, `.cbz`, `.cbr`, `.html`, `.htm`, `.chm`

## Prerequisites

- Linux (Ubuntu/Debian) or macOS
- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally.
- An Ollama model downloaded (e.g., `ministral-3:8b`, `mistral`). To download a model, run in your terminal: `ollama pull ministral-3:8b`.

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

### Detailed Commands

2. **Run the processing**:
   The script can be run without any arguments. By default, it looks for documents in the `source` directory, saves the summaries in the `destination` directory, uses the `ministral-3:8b` model, and applies a summarization level of `7`.
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