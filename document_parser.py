"""Document parser selection with an optional Marker backend."""

from __future__ import annotations

import os

from extractor import extract_document as extract_document_native
from marker_provider import (
    MARKER_SUPPORTED_EXTENSIONS,
    extract_document_with_marker,
    is_marker_available,
)


PARSER_CHOICES = ("native", "marker", "auto")


def extract_document(
    path: str,
    *,
    parser: str = "native",
    marker_mode: str = "fast",
    structured: bool = False,
    source_sha256: str | None = None,
):
    """Extract a document through the selected backend.

    ``auto`` is opt-in. It prefers Marker for supported document formats when
    installed and falls back to the native parser with an explicit diagnostic.
    """
    if parser not in PARSER_CHOICES:
        raise ValueError(f"Unknown parser '{parser}'. Choose from {PARSER_CHOICES}.")

    extension = os.path.splitext(path)[1].lower()
    marker_compatible = extension in MARKER_SUPPORTED_EXTENSIONS

    if parser == "marker":
        if not marker_compatible:
            raise ValueError(
                f"Marker does not support '{extension}' through Dataminder's adapter."
            )
        return extract_document_with_marker(
            path,
            source_sha256=source_sha256,
            mode=marker_mode,
        )

    if parser == "auto" and marker_compatible and is_marker_available():
        try:
            return extract_document_with_marker(
                path,
                source_sha256=source_sha256,
                mode=marker_mode,
            )
        except Exception as exc:
            document = extract_document_native(
                path,
                structured=structured,
                source_sha256=source_sha256,
            )
            document.diagnostics.append(
                {
                    "level": "warning",
                    "code": "marker_fallback",
                    "message": f"Marker failed; native parser used: {exc}",
                }
            )
            document.metadata["parser"] = "native-fallback"
            return document

    document = extract_document_native(
        path,
        structured=structured,
        source_sha256=source_sha256,
    )
    document.metadata["parser"] = "native"
    if parser == "auto" and marker_compatible and not is_marker_available():
        document.diagnostics.append(
            {
                "level": "info",
                "code": "marker_unavailable",
                "message": "Marker is not installed; native parser used.",
            }
        )
    return document
