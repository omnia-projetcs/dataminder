"""Optional adapter from Datalab Marker outputs to Dataminder's DocumentIR."""

from __future__ import annotations

import importlib.util
import re
from typing import Any

from bs4 import BeautifulSoup

from document_ir import DocumentBlock, DocumentIR, create_document


MARKER_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".epub",
}


class MarkerProviderUnavailable(RuntimeError):
    """Raised when the optional Marker runtime is not installed."""


def is_marker_available() -> bool:
    """Return whether the optional ``marker-pdf`` package can be imported."""
    return importlib.util.find_spec("marker") is not None


def _value(item: Any, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _plain_mapping(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value)


def _html_to_markdown(html: str) -> str:
    if not html:
        return ""
    try:
        from markdownify import markdownify

        return markdownify(html, heading_style="ATX").strip()
    except ImportError:
        return BeautifulSoup(html, "html.parser").get_text("\n", strip=True)


def _heading_level(html: str) -> int | None:
    match = re.search(r"<h([1-6])(?:\s|>)", html or "", re.IGNORECASE)
    return int(match.group(1)) if match else None


def document_from_marker_output(
    path: str,
    rendered: Any,
    *,
    source_sha256: str,
    mode: str,
) -> DocumentIR:
    """Map Marker's flat chunk renderer output to Dataminder blocks."""
    document = create_document(path, source_sha256=source_sha256)
    marker_blocks = list(_value(rendered, "blocks", []) or [])
    heading_titles = {}
    heading_levels = {}

    for item in marker_blocks:
        marker_id = str(_value(item, "id", ""))
        block_type = str(_value(item, "block_type", "Text"))
        html = str(_value(item, "html", "") or "")
        if block_type.lower() == "sectionheader":
            heading_titles[marker_id] = _html_to_markdown(html).lstrip("# ").strip()
            level = _heading_level(html)
            if level is not None:
                heading_levels[marker_id] = level

    current_headings: list[str] = []
    blocks = []
    for index, item in enumerate(marker_blocks):
        marker_id = str(_value(item, "id", f"/block/{index}"))
        block_type = str(_value(item, "block_type", "Text"))
        html = str(_value(item, "html", "") or "")
        text = _html_to_markdown(html)
        if not text:
            continue

        hierarchy = _plain_mapping(_value(item, "section_hierarchy", {}))
        heading_path = []
        for level, heading_id in sorted(
            hierarchy.items(),
            key=lambda pair: int(pair[0]) if str(pair[0]).isdigit() else 99,
        ):
            title = heading_titles.get(str(heading_id))
            if title:
                heading_path.append(title)

        if block_type.lower() == "sectionheader":
            title = heading_titles.get(marker_id, text.lstrip("# ").strip())
            level = heading_levels.get(marker_id, 1)
            current_headings[level - 1 :] = [title]
            if not heading_path:
                heading_path = list(current_headings)
        elif not heading_path:
            heading_path = list(current_headings)

        page = _value(item, "page")
        if page is not None:
            page = int(page) + 1  # Marker pages are zero-based.
        bbox = _value(item, "bbox")
        if bbox is not None:
            bbox = [float(coordinate) for coordinate in bbox]

        normalized_type = re.sub(
            r"(?<!^)(?=[A-Z])", "_", block_type
        ).lower()
        blocks.append(
            DocumentBlock(
                id=f"{document.id}/marker{marker_id}",
                block_type=normalized_type,
                text=text,
                page=page,
                bbox=bbox,
                heading_path=heading_path,
                extraction_method=f"marker-{mode}",
                metadata={
                    "marker_id": marker_id,
                    "html": html,
                },
            )
        )

    document.blocks = blocks
    marker_metadata = _value(rendered, "metadata", {})
    document.metadata.update(
        {
            "parser": "marker",
            "marker_mode": mode,
            "extraction_methods": [f"marker-{mode}"] if blocks else [],
            "marker_metadata": _plain_mapping(marker_metadata),
        }
    )
    return document


def extract_document_with_marker(
    path: str,
    *,
    source_sha256: str,
    mode: str = "fast",
) -> DocumentIR:
    """Run Marker locally and convert its chunks to :class:`DocumentIR`.

    LLM correction is intentionally disabled here so selecting Marker does not
    silently send document content to a cloud provider.
    """
    if mode not in {"fast", "balanced"}:
        raise ValueError("Marker mode must be 'fast' or 'balanced'")
    if not is_marker_available():
        raise MarkerProviderUnavailable(
            "Marker is not installed. Install requirements-marker.txt first."
        )

    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config_parser = ConfigParser(
        {
            "output_format": "chunks",
            "mode": mode,
            "use_llm": False,
        }
    )
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=None,
    )
    rendered = converter(path)
    return document_from_marker_output(
        path,
        rendered,
        source_sha256=source_sha256,
        mode=mode,
    )
