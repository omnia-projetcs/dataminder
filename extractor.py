import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from docx import Document

def extract_text_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(path):
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
    try:
        # Requires poppler-utils installed on the system
        images = convert_from_path(path)
        text = ""
        # Requires tesseract-ocr installed on the system
        for img in images:
            text += pytesseract.image_to_string(img, lang='fra+eng') + "\n"
        return text
    except Exception as e:
        print(f"[{path}] Failed to extract text via OCR. Ensure tesseract-ocr and poppler-utils are installed. Error: {e}")
        return ""

def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.txt', '.md']:
        return extract_text_from_txt(path)
    elif ext == '.docx':
        return extract_text_from_docx(path)
    elif ext == '.pdf':
        return extract_text_from_pdf(path)
    else:
        print(f"[{path}] Unsupported file extension: {ext}")
        return ""
