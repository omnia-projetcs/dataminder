import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from docx import Document
import zipfile
import rarfile
from PIL import Image
import subprocess
from pptx import Presentation
import pandas as pd
import tempfile
import shutil
from bs4 import BeautifulSoup
import datetime
import ebooklib
from ebooklib import epub

def log_error(filepath, error_msg):
    with open("error.log", "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] FILE: {filepath} | EXTRACTION ERROR: {error_msg}\n")

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
        log_error(path, f"Failed to extract text via OCR. Ensure tesseract-ocr and poppler-utils are installed. Error: {e}")
        return ""

def extract_text_from_cbz_cbr(path):
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
                        text += pytesseract.image_to_string(img, lang='fra+eng') + "\n"
        elif ext == '.cbr':
            with rarfile.RarFile(path, 'r') as archive:
                image_files = [f for f in archive.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                image_files.sort()
                for img_name in image_files:
                    with archive.open(img_name) as file:
                        img = Image.open(file)
                        text += pytesseract.image_to_string(img, lang='fra+eng') + "\n"
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

def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.txt', '.md']:
        return extract_text_from_txt(path)
    elif ext == '.docx':
        return extract_text_from_docx(path)
    elif ext == '.pdf':
        return extract_text_from_pdf(path)
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
    else:
        log_error(path, f"Unsupported file extension: {ext}")
        return ""
