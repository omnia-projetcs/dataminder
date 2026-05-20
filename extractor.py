import os
import fitz  # PyMuPDF
import zipfile
import rarfile
from PIL import Image
import subprocess
from pptx import Presentation
import pandas as pd
import tempfile
from bs4 import BeautifulSoup
import datetime
import ebooklib
from ebooklib import epub

from ocr_engines import ocr_image, ocr_pdf, structured_parse, is_paddleocr_available

# Module-level OCR configuration (set by main.py based on CLI args)
_ocr_engine = "paddleocr"
_ocr_device = "cpu"
_ocr_lang = "en"


def configure_ocr(engine="paddleocr", device="cpu", lang="en"):
    """
    Configure the OCR engine for all extraction calls.

    Args:
        engine: "paddleocr" (default, deep learning) or "tesseract" (legacy).
        device: "cpu" or "gpu" (PaddleOCR only).
        lang: Language hint (e.g., "en", "fr", "ch").
    """
    global _ocr_engine, _ocr_device, _ocr_lang
    _ocr_engine = engine
    _ocr_device = device
    _ocr_lang = lang

    engine_name = "PaddleOCR PP-OCRv5" if engine == "paddleocr" else "Tesseract"
    print(f"[OCR] Configured: engine={engine_name}, device={device}, lang={lang}")


def log_error(filepath, error_msg):
    with open("error.log", "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] FILE: {filepath} | EXTRACTION ERROR: {error_msg}\n")

def extract_text_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_docx(path):
    from docx import Document
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(path, structured=False):
    """
    Extract text from a PDF file.

    Strategy:
      1. Try direct text extraction with PyMuPDF (for native/digital PDFs)
      2. If text is sparse (scanned PDF), fall back to OCR
      3. If structured=True, use PP-StructureV3 for layout-aware extraction

    Args:
        path: Path to the PDF file.
        structured: If True, use PP-StructureV3 for structured Markdown output.
    """
    # Structured mode: use PP-StructureV3 if available
    if structured:
        result = structured_parse(path, device=_ocr_device)
        if result:
            return result
        print(f"[{path}] PP-StructureV3 unavailable or failed. Falling back to standard extraction.")

    text = ""
    # Try normal PDF extraction first
    try:
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text()
        
        # If the extracted text is very short compared to the number of pages, it's likely a scanned PDF
        if len(text.strip()) < 50 * len(doc):
            print(f"[{path}] PDF appears to be scanned or contains little text. Switching to OCR...")
            text = extract_text_from_pdf_ocr(path)
    except Exception as e:
        print(f"[{path}] Error reading PDF directly: {e}. Trying OCR...")
        text = extract_text_from_pdf_ocr(path)
        
    return text

def extract_text_from_pdf_ocr(path):
    """
    OCR a scanned PDF using the configured OCR engine.

    With PaddleOCR (default):
      - Processes PDF directly (no intermediate image conversion needed)
      - Deep learning text detection + recognition (PP-OCRv5)
      - Automatic orientation correction
      - Automatic distortion/warping correction

    With Tesseract (legacy fallback):
      - Converts pages to images via pdf2image/poppler
      - Traditional LSTM-based OCR
    """
    try:
        return ocr_pdf(path, engine=_ocr_engine, device=_ocr_device, lang=_ocr_lang)
    except Exception as e:
        log_error(path, f"Failed to extract text via OCR ({_ocr_engine}). Error: {e}")
        return ""

def extract_text_from_cbz_cbr(path):
    """
    Extract text from comic book archives (CBZ/CBR) using OCR.

    With PaddleOCR: deep learning detection handles speech bubbles,
    captions, and onomatopoeia much better than Tesseract.
    """
    ext = os.path.splitext(path)[1].lower()
    text = ""
    try:
        if ext == '.cbz':
            with zipfile.ZipFile(path, 'r') as archive:
                image_files = [f for f in archive.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                image_files.sort()
                for img_name in image_files:
                    with archive.open(img_name) as file:
                        img = Image.open(file)
                        text += ocr_image(img, engine=_ocr_engine, device=_ocr_device, lang=_ocr_lang) + "\n"
        elif ext == '.cbr':
            with rarfile.RarFile(path, 'r') as archive:
                image_files = [f for f in archive.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                image_files.sort()
                for img_name in image_files:
                    with archive.open(img_name) as file:
                        img = Image.open(file)
                        text += ocr_image(img, engine=_ocr_engine, device=_ocr_device, lang=_ocr_lang) + "\n"
        return text
    except Exception as e:
        log_error(path, f"Error reading archive (CBZ/CBR): {e}. Make sure 'unrar' is installed.")
        return ""

def extract_text_from_pptx(path):
    try:
        prs = Presentation(path)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    except Exception as e:
        log_error(path, f"Error reading PPTX: {e}")
        return ""

def extract_text_from_doc(path):
    try:
        # Requires antiword installed on the system
        result = subprocess.run(['antiword', path], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        log_error(path, f"Error reading DOC: {e}. Make sure 'antiword' is installed.")
        return ""

def extract_text_from_excel(path):
    try:
        text = ""
        # Read all sheets
        dfs = pd.read_excel(path, sheet_name=None)
        for sheet_name, df in dfs.items():
            text += f"--- Sheet: {sheet_name} ---\n"
            # to_string will convert the dataframe to a readable text table
            text += df.to_string(index=False) + "\n\n"
        return text
    except Exception as e:
        log_error(path, f"Error reading Excel file: {e}")
        return ""

def extract_text_from_html(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        log_error(path, f"Error reading HTML: {e}")
        return ""

def extract_text_from_chm(path):
    try:
        text = ""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Requires libchm-bin installed on the system
            subprocess.run(['extract_chmLib', path, temp_dir], capture_output=True, check=True)
            
            # Find all HTML files in the extracted directory
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.html', '.htm')):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            soup = BeautifulSoup(f, 'html.parser')
                            extracted = soup.get_text(separator='\n', strip=True)
                            if extracted:
                                text += f"--- Section: {file} ---\n" + extracted + "\n\n"
        return text
    except Exception as e:
        log_error(path, f"Error reading CHM: {e}. Make sure 'libchm-bin' is installed.")
        return ""

def extract_text_from_epub(path):
    try:
        book = epub.read_epub(path, options={'ignore_ncx': True})
        text = ""
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            extracted = soup.get_text(separator='\n', strip=True)
            if extracted:
                text += extracted + "\n\n"
        return text
    except Exception as e:
        log_error(path, f"Error reading EPUB: {e}")
        return ""

def extract_text_from_image(path):
    """
    Extract text from a standalone image file using OCR.

    Supports: PNG, JPG, JPEG, WEBP, BMP, TIFF.
    """
    try:
        img = Image.open(path)
        return ocr_image(img, engine=_ocr_engine, device=_ocr_device, lang=_ocr_lang)
    except Exception as e:
        log_error(path, f"Error reading image for OCR: {e}")
        return ""

def extract_text(path, structured=False):
    """
    Extract text from a document file.

    Args:
        path: Path to the document file.
        structured: If True and the file is a PDF, use PP-StructureV3
                    for layout-aware structured Markdown output.
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.txt', '.md', '.rst']:
        return extract_text_from_txt(path)
    elif ext == '.docx':
        return extract_text_from_docx(path)
    elif ext == '.pdf':
        return extract_text_from_pdf(path, structured=structured)
    elif ext in ['.cbz', '.cbr']:
        return extract_text_from_cbz_cbr(path)
    elif ext == '.pptx':
        return extract_text_from_pptx(path)
    elif ext == '.doc':
        return extract_text_from_doc(path)
    elif ext in ['.xls', '.xlsx']:
        return extract_text_from_excel(path)
    elif ext in ['.html', '.htm']:
        return extract_text_from_html(path)
    elif ext == '.chm':
        return extract_text_from_chm(path)
    elif ext == '.epub':
        return extract_text_from_epub(path)
    elif ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif']:
        return extract_text_from_image(path)
    else:
        log_error(path, f"Unsupported file extension: {ext}")
        return ""
