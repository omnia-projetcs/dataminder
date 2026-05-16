# Dataminder: Document Extraction and Summarization with Ollama

Dataminder is a Python tool that automatically extracts text from various types of documents (Text, Markdown, Word, standard PDFs, and scanned PDFs) and generates a detailed summary in Markdown format using a local Artificial Intelligence model (Ollama).

## Features
- Support for multiple formats: `.txt`, `.md`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.pptx`, `.pdf`, `.cbz`, `.cbr`, `.html`, `.htm`, `.chm`.
- **Automatic OCR (Optical Character Recognition)**: If a scanned PDF is detected (very little raw text), the script switches to OCR to read the content from the images.
- Seamless integration with Ollama for summary generation.

## Prerequisites

- Linux (Ubuntu/Debian for the installation script)
- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally.
- An Ollama model downloaded (e.g., `ministral-3:8b`, `mistral`). To download a model, run in your terminal: `ollama pull ministral-3:8b`.

## Installation

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

### Detailed Commands

2. **Run the processing**:
   The script can be run without any arguments. By default, it looks for documents in the `source` directory, saves the summaries in the `destination` directory, and uses the `ministral-3:8b` model.
   ```bash
   python main.py
   ```
   
   You can customize this behavior using command line arguments:
   ```bash
   python main.py --source ./my_documents --dest ./my_summaries --model llama3
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