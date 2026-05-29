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

from PIL import Image
from logger import log_error as _shared_log_error

# ---------------------------------------------------------------------------
# Lazy-loaded singletons — heavy imports only when actually needed
# ---------------------------------------------------------------------------
_paddleocr_instance = None
_paddleocr_config = None  # (device, lang) tuple to detect config changes
_paddleocr_structure_instance = None
_tesseract_available = None
_paddleocr_available = None


def _log_error(filepath, error_msg):
    _shared_log_error(filepath, error_msg, category="OCR ENGINE")


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
    Reinitializes if device or lang settings change.

    Args:
        device: "cpu" or "gpu" for inference device.
        lang: Language hint ("en", "fr", "ch", etc.). Default "en".
              PP-OCRv5 handles multilingual natively so this is a hint.
    """
    global _paddleocr_instance, _paddleocr_config
    current_config = (device, lang)
    if _paddleocr_instance is None or _paddleocr_config != current_config:
        from paddleocr import PaddleOCR
        _paddleocr_instance = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=True,  # Auto-correct orientation
            use_doc_unwarping=True,             # Auto-correct distortion/warping
            use_textline_orientation=True,       # Correct text line direction
            device=device,
            enable_mkldnn=False,
        )
        _paddleocr_config = current_config
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


def paddleocr_pdf_to_string(pdf_path, device="cpu", lang="en", dpi=200, max_pages=0):
    """
    Run PaddleOCR on a PDF file and return all extracted text.

    Processes page-by-page to avoid OOM on large scanned PDFs.
    Each page is rendered to an image via PyMuPDF, OCR'd, then discarded.

    Args:
        pdf_path: Path to the PDF file.
        device: "cpu" or "gpu".
        lang: Language hint for PaddleOCR.
        dpi: Resolution for rendering pages (default 200). Lower = less RAM.
        max_pages: Maximum pages to process (0 = unlimited).

    Returns:
        Extracted text from all pages as a single string.
    """
    import fitz
    import numpy as np

    ocr = _get_paddleocr(device=device, lang=lang)
    all_text = []

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_process = total_pages if max_pages <= 0 else min(total_pages, max_pages)

        if max_pages > 0 and total_pages > max_pages:
            print(f"[PaddleOCR] PDF has {total_pages} pages, limiting to {max_pages} (use --ocr-max-pages to change)")

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_num in range(pages_to_process):
            try:
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_np = np.array(img)
                # Release pixmap memory immediately
                del pix

                results = ocr.predict(img_np)
                del img_np, img

                for res in results:
                    if hasattr(res, 'rec_texts') and res.rec_texts:
                        for text in res.rec_texts:
                            if text and text.strip():
                                all_text.append(text.strip())
                    elif hasattr(res, 'text') and res.text:
                        all_text.append(res.text.strip())
            except Exception as e:
                print(f"[PaddleOCR] Failed on page {page_num + 1}/{pages_to_process}: {e}")
                continue

        doc.close()
    except Exception as e:
        _log_error(pdf_path, f"PaddleOCR PDF processing failed: {e}")

    return "\n".join(all_text)


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


def ocr_pdf(pdf_path, engine="tesseract", device="cpu", lang="en", dpi=200, max_pages=0):
    """
    Unified OCR interface: run OCR on a scanned PDF.

    Tries the requested engine, falls back to the other if it fails or returns no text.
    Processes pages one at a time to prevent OOM on large scanned PDFs.

    Args:
        pdf_path: Path to the PDF file.
        engine: "paddleocr" or "tesseract".
        device: "cpu" or "gpu" (only for paddleocr).
        lang: Language hint.
        dpi: Resolution for page rendering (default 200). Lower = less RAM.
        max_pages: Maximum pages to process (0 = unlimited).

    Returns:
        Extracted text as a string.
    """
    text = ""
    if engine == "paddleocr":
        if is_paddleocr_available():
            try:
                text = paddleocr_pdf_to_string(pdf_path, device=device, lang=lang, dpi=dpi, max_pages=max_pages)
            except Exception as e:
                print(f"[OCR] PaddleOCR PDF extraction failed: {e}")
        else:
            print("[OCR] PaddleOCR not available for PDF.")
            
        if not text.strip():
            print("[OCR] PaddleOCR failed or yielded no text. Switching to Tesseract as fallback...")
            if is_tesseract_available():
                try:
                    text = _tesseract_pdf_fallback(pdf_path, lang, dpi=dpi, max_pages=max_pages)
                except Exception as e:
                    print(f"[OCR] Tesseract fallback PDF extraction failed: {e}")
            else:
                print("[OCR] Tesseract fallback not available.")
                
    elif engine == "tesseract":
        if is_tesseract_available():
            try:
                text = _tesseract_pdf_fallback(pdf_path, lang, dpi=dpi, max_pages=max_pages)
            except Exception as e:
                print(f"[OCR] Tesseract PDF extraction failed: {e}")
        else:
            print("[OCR] Tesseract not available.")
            
        if not text.strip():
            print("[OCR] Tesseract failed or yielded no text. Switching to PaddleOCR as fallback...")
            if is_paddleocr_available():
                try:
                    text = paddleocr_pdf_to_string(pdf_path, device=device, lang=lang, dpi=dpi, max_pages=max_pages)
                except Exception as e:
                    print(f"[OCR] PaddleOCR fallback PDF extraction failed: {e}")
            else:
                print("[OCR] PaddleOCR fallback not available.")
    else:
        raise ValueError(f"Unknown OCR engine: {engine}")
        
    return text



def _tesseract_pdf_fallback(pdf_path, lang, dpi=200, max_pages=0):
    """
    Convert PDF to images and OCR with Tesseract (legacy method).

    Processes page-by-page using PyMuPDF to avoid loading all images into
    memory at once (which caused OOM kills on large scanned PDFs).
    """
    if not is_tesseract_available():
        _log_error(pdf_path, "Tesseract not available for PDF OCR fallback")
        return ""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_process = total_pages if max_pages <= 0 else min(total_pages, max_pages)

        if max_pages > 0 and total_pages > max_pages:
            print(f"[Tesseract] PDF has {total_pages} pages, limiting to {max_pages} (use --ocr-max-pages to change)")

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        tess_lang = _lang_to_tesseract(lang)
        text_parts = []

        for page_num in range(pages_to_process):
            try:
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                del pix  # Release pixmap memory immediately

                page_text = tesseract_image_to_string(img, lang=tess_lang)
                del img  # Release image memory immediately

                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                print(f"[Tesseract] Failed on page {page_num + 1}/{pages_to_process}: {e}")
                continue

        doc.close()
        return "\n".join(text_parts)
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
