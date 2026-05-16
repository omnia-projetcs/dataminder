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
- An Ollama model downloaded (e.g., `llama3`, `mistral`). To download a model, run in your terminal: `ollama run llama3`.

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

2. **Working directories**:
   The script works with two folders:
   - `input_docs/`: Place all the documents you want to summarize here.
   - `output_summaries/`: This is where the AI-generated summaries (in `.md` format) will be saved.

   *(If they do not exist, the script will create them on the first run).*

3. **Run the processing**:
   ```bash
   python main.py
   ```

## Configuration

By default, the script uses the **`llama3`** model.
If you wish to use another model (e.g., `mistral` or `phi3`), open the `main.py` file and modify the configuration variable at the top of the file:

```python
OLLAMA_MODEL = "llama3"
```