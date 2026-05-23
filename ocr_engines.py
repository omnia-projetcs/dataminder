"""
OCR Engine abstraction for Dataminder.

Supports two backends:
  - tesseract : Legacy Tesseract OCR (default)
  - paddleocr : PaddleOCR PP-OCRv5 (deep learning, much higher accuracy, enabled via options)

PaddleOCR features integrated from https://github.com/PaddlePaddle/PaddleOCR:
  - PP-OCRv5 text detection + recognition (109 languages, 13% accuracy boost)
  - Automatic document orientation correction
  - Automatic document distortion/warping correction
  - Text line orientation classification
  - Optional GPU acceleration
  - Optional structured document parsing (PP-StructureV3)
"""

import os
# Disable oneDNN (MKLDNN) to prevent NotImplementedError in PaddlePaddle 3.3.0+ CPU inference
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import datetime
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy-loaded singletons — heavy imports only when actually needed
# ---------------------------------------------------------------------------
_paddleocr_instance = None
_paddleocr_structure_instance = None
_tesseract_available = None
_paddleocr_available = None


def _log_error(filepath, error_msg):
    with open("error.log", "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] FILE: {filepath} | OCR ENGINE ERROR: {error_msg}\n")


# ---------------------------------------------------------------------------
# Engine availability checks
# ---------------------------------------------------------------------------

def is_paddleocr_available():
    """Check if PaddleOCR is installed and importable."""
    global _paddleocr_available
    if _paddleocr_available is None:
        try:
            import paddle  # noqa: F401
            from paddleocr import PaddleOCR  # noqa: F401
            _paddleocr_available = True
        except ImportError:
            _paddleocr_available = False
    return _paddleocr_available


def is_tesseract_available():
    """Check if Tesseract is installed and importable."""
    global _tesseract_available
    if _tesseract_available is None:
        try:
            import pytesseract  # noqa: F401
            # Quick check that the binary is accessible
            pytesseract.get_tesseract_version()
            _tesseract_available = True
        except Exception:
            _tesseract_available = False
    return _tesseract_available


# ---------------------------------------------------------------------------
# PaddleOCR Engine
# ---------------------------------------------------------------------------

def _get_paddleocr(device="cpu", lang="en"):
    """
    Get or create a singleton PaddleOCR instance.

    Uses PP-OCRv5 server models for maximum accuracy. The instance is
    created lazily on first call and reused for subsequent calls.

    Args:
        device: "cpu" or "gpu" for inference device.
        lang: Language hint ("en", "fr", "ch", etc.). Default "en".
              PP-OCRv5 handles multilingual natively so this is a hint.
    """
    global _paddleocr_instance
    if _paddleocr_instance is None:
        from paddleocr import PaddleOCR
        _paddleocr_instance = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=True,  # Auto-correct orientation
            use_doc_unwarping=True,             # Auto-correct distortion/warping
            use_textline_orientation=True,       # Correct text line direction
            device=device,
            enable_mkldnn=False,
        )
        print(f"[PaddleOCR] Initialized PP-OCRv5 engine (device={device}, lang={lang})")
    return _paddleocr_instance


def paddleocr_image_to_string(image, device="cpu", lang="en"):
    """
    Run PaddleOCR on a PIL Image and return extracted text as a string.

    This replaces pytesseract.image_to_string() with superior deep-learning
    based text detection + recognition from PP-OCRv5.

    Features over Tesseract:
      - Deep learning text detection (finds text zones precisely)
      - PP-OCRv5 recognition (13% more accurate than v4)
      - Automatic orientation correction
      - Automatic distortion/warping correction
      - 109 languages supported natively

    Args:
        image: PIL Image object or path to image file.
        device: "cpu" or "gpu".
        lang: Language hint for PaddleOCR.

    Returns:
        Extracted text as a single string.
    """
    ocr = _get_paddleocr(device=device, lang=lang)

    # PaddleOCR can accept file paths or numpy arrays
    if isinstance(image, str):
        input_data = image
    elif isinstance(image, Image.Image):
        import numpy as np
        input_data = np.array(image)
    else:
        input_data = image

    try:
        results = ocr.predict(input_data)
        lines = []
        for res in results:
            # res contains detected text boxes with recognized text
            if hasattr(res, 'rec_texts') and res.rec_texts:
                for text in res.rec_texts:
                    if text and text.strip():
                        lines.append(text.strip())
            elif hasattr(res, 'text') and res.text:
                lines.append(res.text.strip())
        return "\n".join(lines)
    except Exception as e:
        _log_error("<image>", f"PaddleOCR prediction failed: {e}")
        return ""


def paddleocr_pdf_to_string(pdf_path, device="cpu", lang="en"):
    """
    Run PaddleOCR on a PDF file and return all extracted text.

    PaddleOCR can process PDF pages directly when given a file path,
    handling multi-page documents automatically.

    Args:
        pdf_path: Path to the PDF file.
        device: "cpu" or "gpu".
        lang: Language hint for PaddleOCR.

    Returns:
        Extracted text from all pages as a single string.
    """
    ocr = _get_paddleocr(device=device, lang=lang)

    try:
        results = ocr.predict(pdf_path)
        all_text = []
        for page_res in results:
            if hasattr(page_res, 'rec_texts') and page_res.rec_texts:
                for text in page_res.rec_texts:
                    if text and text.strip():
                        all_text.append(text.strip())
            elif hasattr(page_res, 'text') and page_res.text:
                all_text.append(page_res.text.strip())
        return "\n".join(all_text)
    except Exception as e:
        _log_error(pdf_path, f"PaddleOCR PDF processing failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# PP-StructureV3 — Structured Document Parsing
# ---------------------------------------------------------------------------

def _get_paddleocr_structure(device="cpu"):
    """
    Get or create a PP-StructureV3 instance for structured document parsing.

    PP-StructureV3 extracts:
      - Document layout (titles, paragraphs, tables, images, formulas)
      - Tables → HTML/Markdown
      - Formulas → LaTeX
      - Full document → structured Markdown
    """
    global _paddleocr_structure_instance
    if _paddleocr_structure_instance is None:
        try:
            from paddleocr import PPStructureV3
            _paddleocr_structure_instance = PPStructureV3(device=device)
            print(f"[PaddleOCR] Initialized PP-StructureV3 engine (device={device})")
        except ImportError:
            # PP-StructureV3 may require additional dependencies
            print("[PaddleOCR] PP-StructureV3 not available. Install with: pip install 'paddleocr[structure]'")
            return None
    return _paddleocr_structure_instance


def structured_parse(file_path, device="cpu"):
    """
    Parse a document using PP-StructureV3 and return structured Markdown.

    This extracts layout-aware content: headings, paragraphs, tables (as
    Markdown tables), and formulas (as LaTeX). Much better than flat OCR
    for complex documents.

    Args:
        file_path: Path to PDF or image file.
        device: "cpu" or "gpu".

    Returns:
        Structured Markdown string, or None if PP-StructureV3 is unavailable.
    """
    structure = _get_paddleocr_structure(device=device)
    if structure is None:
        return None

    try:
        results = structure.predict(file_path)
        markdown_parts = []
        for page_res in results:
            if hasattr(page_res, 'markdown') and page_res.markdown:
                markdown_parts.append(page_res.markdown)
            elif hasattr(page_res, 'text') and page_res.text:
                markdown_parts.append(page_res.text)
        return "\n\n".join(markdown_parts) if markdown_parts else None
    except Exception as e:
        _log_error(file_path, f"PP-StructureV3 parsing failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Tesseract Engine (legacy fallback)
# ---------------------------------------------------------------------------

def tesseract_image_to_string(image, lang='fra+eng'):
    """
    Run Tesseract OCR on a PIL Image. Legacy fallback.

    Args:
        image: PIL Image object.
        lang: Tesseract language codes (default: 'fra+eng').

    Returns:
        Extracted text as a string.
    """
    import pytesseract
    try:
        return pytesseract.image_to_string(image, lang=lang)
    except Exception as e:
        _log_error("<image>", f"Tesseract OCR failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Unified OCR Interface
# ---------------------------------------------------------------------------

def ocr_image(image, engine="tesseract", device="cpu", lang="en"):
    """
    Unified OCR interface: run OCR on a PIL Image.

    Tries the requested engine, falls back to the other if it fails or returns no text.

    Args:
        image: PIL Image object or path to image.
        engine: "paddleocr" or "tesseract".
        device: "cpu" or "gpu" (only for paddleocr).
        lang: Language hint.

    Returns:
        Extracted text as a string.
    """
    text = ""
    if engine == "paddleocr":
        if is_paddleocr_available():
            try:
                text = paddleocr_image_to_string(image, device=device, lang=lang)
            except Exception as e:
                print(f"[OCR] PaddleOCR image extraction failed: {e}")
        else:
            print("[OCR] PaddleOCR not available.")
            
        if not text.strip():
            print("[OCR] PaddleOCR failed or yielded no text. Switching to Tesseract as fallback...")
            if is_tesseract_available():
                try:
                    tess_lang = _lang_to_tesseract(lang)
                    text = tesseract_image_to_string(image, lang=tess_lang)
                except Exception as e:
                    print(f"[OCR] Tesseract fallback image extraction failed: {e}")
            else:
                print("[OCR] Tesseract fallback not available.")
                
    elif engine == "tesseract":
        if is_tesseract_available():
            try:
                tess_lang = _lang_to_tesseract(lang)
                text = tesseract_image_to_string(image, lang=tess_lang)
            except Exception as e:
                print(f"[OCR] Tesseract image extraction failed: {e}")
        else:
            print("[OCR] Tesseract not available.")
            
        if not text.strip():
            print("[OCR] Tesseract failed or yielded no text. Switching to PaddleOCR as fallback...")
            if is_paddleocr_available():
                try:
                    text = paddleocr_image_to_string(image, device=device, lang=lang)
                except Exception as e:
                    print(f"[OCR] PaddleOCR fallback image extraction failed: {e}")
            else:
                print("[OCR] PaddleOCR fallback not available.")
    else:
        raise ValueError(f"Unknown OCR engine: {engine}. Use 'paddleocr' or 'tesseract'.")

    return text


def ocr_pdf(pdf_path, engine="tesseract", device="cpu", lang="en"):
    """
    Unified OCR interface: run OCR on a scanned PDF.

    Tries the requested engine, falls back to the other if it fails or returns no text.

    Args:
        pdf_path: Path to the PDF file.
        engine: "paddleocr" or "tesseract".
        device: "cpu" or "gpu" (only for paddleocr).
        lang: Language hint.

    Returns:
        Extracted text as a string.
    """
    text = ""
    if engine == "paddleocr":
        if is_paddleocr_available():
            try:
                text = paddleocr_pdf_to_string(pdf_path, device=device, lang=lang)
            except Exception as e:
                print(f"[OCR] PaddleOCR PDF extraction failed: {e}")
        else:
            print("[OCR] PaddleOCR not available for PDF.")
            
        if not text.strip():
            print("[OCR] PaddleOCR failed or yielded no text. Switching to Tesseract as fallback...")
            if is_tesseract_available():
                try:
                    text = _tesseract_pdf_fallback(pdf_path, lang)
                except Exception as e:
                    print(f"[OCR] Tesseract fallback PDF extraction failed: {e}")
            else:
                print("[OCR] Tesseract fallback not available.")
                
    elif engine == "tesseract":
        if is_tesseract_available():
            try:
                text = _tesseract_pdf_fallback(pdf_path, lang)
            except Exception as e:
                print(f"[OCR] Tesseract PDF extraction failed: {e}")
        else:
            print("[OCR] Tesseract not available.")
            
        if not text.strip():
            print("[OCR] Tesseract failed or yielded no text. Switching to PaddleOCR as fallback...")
            if is_paddleocr_available():
                try:
                    text = paddleocr_pdf_to_string(pdf_path, device=device, lang=lang)
                except Exception as e:
                    print(f"[OCR] PaddleOCR fallback PDF extraction failed: {e}")
            else:
                print("[OCR] PaddleOCR fallback not available.")
    else:
        raise ValueError(f"Unknown OCR engine: {engine}")
        
    return text



def _tesseract_pdf_fallback(pdf_path, lang):
    """Convert PDF to images and OCR with Tesseract (legacy method)."""
    if not is_tesseract_available():
        _log_error(pdf_path, "Tesseract not available for PDF OCR fallback")
        return ""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path)
        text = ""
        tess_lang = _lang_to_tesseract(lang)
        for img in images:
            text += tesseract_image_to_string(img, lang=tess_lang) + "\n"
        return text
    except Exception as e:
        _log_error(pdf_path, f"Tesseract PDF fallback failed: {e}")
        return ""


def _lang_to_tesseract(lang):
    """Convert a PaddleOCR-style language code to Tesseract format."""
    mapping = {
        "en": "eng",
        "fr": "fra",
        "ch": "chi_sim",
        "de": "deu",
        "es": "spa",
        "it": "ita",
        "pt": "por",
        "ja": "jpn",
        "ko": "kor",
        "ar": "ara",
        "ru": "rus",
    }
    if lang in mapping:
        return mapping[lang]
    # Default: try fra+eng for best coverage
    return "fra+eng"
