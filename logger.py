"""
Centralized error logging for Dataminder.

All modules use this shared function to write errors to a single
log file inside the data/ directory, keeping the project root clean.
"""

import os
import datetime

# All logs and cache live under data/ relative to this file's location
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ERROR_LOG_PATH = os.path.join(DATA_DIR, "error.log")


def log_error(filepath, error_msg, category="GENERAL"):
    """
    Append an error entry to data/error.log.

    Args:
        filepath: The file that caused the error.
        error_msg: Description of the error.
        category: Error category tag (e.g., PIPELINE, QA GENERATION, EXTRACTION, OCR ENGINE, TRANSCRIPTION).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] FILE: {filepath} | {category} ERROR: {error_msg}\n")
