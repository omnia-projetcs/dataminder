"""Structured intermediate representation used by the ingestion pipeline.

The IR deliberately stays independent from any OCR or LLM implementation.  An
extractor only has to produce :class:`DocumentBlock` objects; renderers and
downstream dataset builders can then preserve provenance without knowing which
engine created them.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def sha256_file(path: str, buffer_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(buffer_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class DocumentBlock:
    """A traceable unit of extracted document content."""

    id: str
    block_type: str
    text: str
    page: int | None = None
    bbox: list[float] | None = None
    heading_path: list[str] = field(default_factory=list)
    extraction_method: str = "unknown"
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentIR:
    """Engine-neutral representation of one source document."""

    id: str
    source_path: str
    source_name: str
    source_sha256: str
    media_type: str
    blocks: list[DocumentBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(block.text.strip() for block in self.blocks if block.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "id": self.id,
            "source_path": self.source_path,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "media_type": self.media_type,
            "metadata": self.metadata,
            "diagnostics": self.diagnostics,
            "blocks": [block.to_dict() for block in self.blocks],
        }


def create_document(
    path: str,
    blocks: Iterable[DocumentBlock] | None = None,
    *,
    source_sha256: str | None = None,
    metadata: dict[str, Any] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> DocumentIR:
    """Create a :class:`DocumentIR` with a content-addressed identifier."""
    source_sha256 = source_sha256 or sha256_file(path)
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return DocumentIR(
        id=f"sha256:{source_sha256}",
        source_path=os.path.abspath(path),
        source_name=os.path.basename(path),
        source_sha256=source_sha256,
        media_type=media_type,
        blocks=list(blocks or []),
        metadata=dict(metadata or {}),
        diagnostics=list(diagnostics or []),
    )


def blocks_from_text(
    text: str,
    document_id: str,
    *,
    extraction_method: str,
    page: int | None = None,
    id_prefix: str | None = None,
) -> list[DocumentBlock]:
    """Turn Markdown/plain text into paragraph-level blocks.

    Markdown headings update ``heading_path`` and fenced code is kept intact.
    This is intentionally conservative: format-specific providers can later
    replace it with richer blocks while keeping the same public contract.
    """
    if not text or not text.strip():
        return []

    raw_units: list[tuple[str, str, list[str]]] = []
    heading_stack: list[str] = []
    paragraph: list[str] = []
    code_lines: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        if not paragraph:
            return
        content = "\n".join(paragraph).strip()
        paragraph.clear()
        if content:
            block_type = "table" if _looks_like_table(content) else "text"
            raw_units.append((block_type, content, list(heading_stack)))

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if in_code:
                code_lines.append(line)
                raw_units.append(("code", "\n".join(code_lines).strip(), list(heading_stack)))
                code_lines.clear()
                in_code = False
            else:
                flush_paragraph()
                in_code = True
                code_lines.append(line)
            continue

        if in_code:
            code_lines.append(line)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack[level - 1 :] = [title]
            raw_units.append(("section_header", line.strip(), list(heading_stack)))
        elif line.strip():
            paragraph.append(line)
        else:
            flush_paragraph()

    flush_paragraph()
    if code_lines:
        raw_units.append(("code", "\n".join(code_lines).strip(), list(heading_stack)))

    blocks = []
    id_prefix = id_prefix or document_id
    for index, (block_type, content, heading_path) in enumerate(raw_units):
        blocks.append(
            DocumentBlock(
                id=f"{id_prefix}/block/{index}",
                block_type=block_type,
                text=content,
                page=page,
                heading_path=heading_path,
                extraction_method=extraction_method,
            )
        )
    return blocks


def _looks_like_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) >= 2 and all(line.count("|") >= 2 for line in lines[:3])


def _split_large_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split one oversized block at paragraph/line boundaries."""
    parts = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = text.rfind("\n\n", start, end)
            if split_at <= start:
                split_at = text.rfind("\n", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            end = split_at if split_at > start else end
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap_chars)
        start = next_start
    return parts


def build_chunks(
    document: DocumentIR,
    *,
    max_chars: int = 5000,
    overlap_chars: int = 300,
) -> list[dict[str, Any]]:
    """Build RAG-ready chunks while retaining block and page provenance."""
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than max_chars")

    fragments: list[tuple[str, DocumentBlock]] = []
    for block in document.blocks:
        text = block.text.strip()
        if not text:
            continue
        for part in _split_large_text(text, max_chars, overlap_chars):
            fragments.append((part, block))

    chunks: list[dict[str, Any]] = []
    current_texts: list[str] = []
    current_blocks: list[DocumentBlock] = []

    def flush() -> None:
        if not current_texts:
            return
        text = "\n\n".join(current_texts).strip()
        block_ids = list(dict.fromkeys(block.id for block in current_blocks))
        pages = sorted({block.page for block in current_blocks if block.page is not None})
        block_types = list(dict.fromkeys(block.block_type for block in current_blocks))
        heading_path = next(
            (block.heading_path for block in reversed(current_blocks) if block.heading_path),
            [],
        )
        extraction_methods = list(
            dict.fromkeys(block.extraction_method for block in current_blocks)
        )
        chunk_index = len(chunks)
        chunk_digest = hashlib.sha256(
            f"{document.id}:{chunk_index}:{text}".encode("utf-8")
        ).hexdigest()
        chunks.append(
            {
                "schema_version": 1,
                "id": f"sha256:{chunk_digest}",
                "document_id": document.id,
                "source_path": document.metadata.get(
                    "source_relative_path", document.source_path
                ),
                "source_name": document.source_name,
                "text": text,
                "block_ids": block_ids,
                "block_types": block_types,
                "pages": pages,
                "heading_path": heading_path,
                "extraction_methods": extraction_methods,
                "char_count": len(text),
            }
        )
        current_texts.clear()
        current_blocks.clear()

    for fragment, block in fragments:
        projected = len(fragment) + sum(len(item) for item in current_texts)
        projected += max(0, len(current_texts)) * 2
        if current_texts and projected > max_chars:
            flush()
        current_texts.append(fragment)
        current_blocks.append(block)
    flush()
    return chunks


def chunks_to_jsonl(chunks: Iterable[dict[str, Any]]) -> str:
    """Serialize chunks deterministically as newline-delimited JSON."""
    lines = [json.dumps(chunk, ensure_ascii=False, sort_keys=True) for chunk in chunks]
    return "\n".join(lines) + ("\n" if lines else "")
