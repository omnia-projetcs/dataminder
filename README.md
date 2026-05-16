# Dataminder: Document Extraction and Summarization with Ollama

Dataminder is a Python tool that automatically extracts text from various types of documents (Text, Markdown, Word, standard PDFs, and scanned PDFs) and generates a detailed summary in Markdown format using a local Artificial Intelligence model (Ollama).

## Features
- Support for multiple formats: `.txt`, `.md`, `.docx`, `.pdf`.
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

2. **Run the processing**:
   The script can be run without any arguments. By default, it looks for documents in the `source` directory, saves the summaries in the `destination` directory, and uses the `ministral-3:8b` model.
   ```bash
   python main.py
   ```
   
   You can customize this behavior using command line arguments:
   ```bash
   python main.py --source ./my_documents --dest ./my_summaries --model llama3
   ```
   
   The output files will be named with the current date and time to ensure uniqueness (e.g., `2026-05-16_12-30-00_document.md`).

## Configuration

You can change the AI model directly via the `--model` parameter when launching the script:

```bash
python main.py --model mistral
```