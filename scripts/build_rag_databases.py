#!/usr/bin/env python3
"""Build two standalone, provenance-preserving SQLite RAG databases."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import fitz
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {
    ".pdf", ".epub", ".md", ".rst", ".txt", ".doc", ".pptx",
}
SPACE_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{3,}")
HEADING_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s+|={2,}\s*)(.+?)\s*$")
WORD_RE = re.compile(r"\b[\wÀ-ÿ'-]+\b", re.UNICODE)


@dataclass
class TextUnit:
    text: str
    page: int | None = None
    section: str = ""
    method: str = "unknown"


@dataclass
class ExtractedDocument:
    title: str
    units: list[TextUnit]
    page_count: int | None
    parser: str
    diagnostics: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: str) -> str:
    value = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [SPACE_RE.sub(" ", line).strip() for line in value.splitlines()]
    return BLANK_RE.sub("\n\n", "\n".join(lines)).strip()


def canonical_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def repeated_pdf_lines(pages: list[str]) -> set[str]:
    occurrences: collections.Counter[str] = collections.Counter()
    for text in pages:
        page_lines = {
            canonical_text(line)
            for line in text.splitlines()
            if 3 <= len(canonical_text(line)) <= 120
        }
        occurrences.update(page_lines)
    threshold = max(3, math.ceil(len(pages) * 0.20))
    return {line for line, count in occurrences.items() if count >= threshold}


def extract_pdf(path: Path) -> ExtractedDocument:
    diagnostics = []
    units = []
    with fitz.open(path) as document:
        metadata = document.metadata or {}
        title = clean_text(metadata.get("title") or "") or path.stem
        raw_pages = []
        for page in document:
            try:
                raw_pages.append(page.get_text("text", sort=True))
            except Exception as exc:
                diagnostics.append(f"page_{page.number + 1}: {exc}")
                raw_pages.append("")
        boilerplate = repeated_pdf_lines(raw_pages)
        for page_number, raw_text in enumerate(raw_pages, 1):
            retained = [
                line
                for line in raw_text.splitlines()
                if canonical_text(line) not in boilerplate
            ]
            text = clean_text("\n".join(retained))
            if text:
                units.append(
                    TextUnit(
                        text=text,
                        page=page_number,
                        method="pymupdf-native",
                    )
                )
        return ExtractedDocument(
            title=title,
            units=units,
            page_count=len(document),
            parser="pymupdf-native",
            diagnostics=diagnostics,
        )


def epub_spine(zip_handle: zipfile.ZipFile) -> list[tuple[str, str]]:
    try:
        container = ElementTree.fromstring(
            zip_handle.read("META-INF/container.xml")
        )
        rootfile = next(
            node.attrib["full-path"]
            for node in container.iter()
            if node.tag.endswith("rootfile")
        )
        opf = ElementTree.fromstring(zip_handle.read(rootfile))
        base = Path(rootfile).parent
        manifest = {
            node.attrib["id"]: node.attrib["href"]
            for node in opf.iter()
            if node.tag.endswith("item")
            and node.attrib.get("media-type") in {
                "application/xhtml+xml",
                "text/html",
            }
        }
        ordered = []
        for node in opf.iter():
            if node.tag.endswith("itemref"):
                href = manifest.get(node.attrib.get("idref", ""))
                if href:
                    ordered.append((node.attrib.get("idref", ""), str(base / href)))
        if ordered:
            return ordered
    except Exception:
        pass
    return [
        (Path(name).stem, name)
        for name in sorted(zip_handle.namelist())
        if name.casefold().endswith((".xhtml", ".html", ".htm"))
    ]


def extract_epub(path: Path) -> ExtractedDocument:
    units = []
    diagnostics = []
    title = path.stem
    with zipfile.ZipFile(path) as archive:
        for item_id, member in epub_spine(archive):
            try:
                soup = BeautifulSoup(archive.read(member), "html.parser")
            except Exception as exc:
                diagnostics.append(f"{member}: {exc}")
                continue
            heading = soup.find(["h1", "h2", "h3", "title"])
            section = clean_text(heading.get_text(" ", strip=True)) if heading else item_id
            text = clean_text(soup.get_text("\n", strip=True))
            if text:
                units.append(
                    TextUnit(
                        text=text,
                        section=section,
                        method="epub-zip-beautifulsoup",
                    )
                )
    return ExtractedDocument(
        title=title,
        units=units,
        page_count=None,
        parser="epub-zip-beautifulsoup",
        diagnostics=diagnostics,
    )


def split_markdown_sections(text: str, method: str) -> list[TextUnit]:
    units = []
    current = []
    section = ""
    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if heading and current:
            content = clean_text("\n".join(current))
            if content:
                units.append(TextUnit(content, section=section, method=method))
            current = []
        if heading:
            section = clean_text(heading.group(1))
        current.append(line)
    content = clean_text("\n".join(current))
    if content:
        units.append(TextUnit(content, section=section, method=method))
    return units


def extract_plain(path: Path) -> ExtractedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    return ExtractedDocument(
        title=path.stem,
        units=split_markdown_sections(text, "plain-text"),
        page_count=None,
        parser="plain-text",
        diagnostics=[],
    )


def extract_doc(path: Path) -> ExtractedDocument:
    result = subprocess.run(
        ["antiword", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return ExtractedDocument(
        title=path.stem,
        units=[TextUnit(clean_text(result.stdout), method="antiword")],
        page_count=None,
        parser="antiword",
        diagnostics=[],
    )


def extract_pptx(path: Path) -> ExtractedDocument:
    units = []
    with zipfile.ZipFile(path) as archive:
        slides = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        )
        for slide_number, member in enumerate(slides, 1):
            root = ElementTree.fromstring(archive.read(member))
            text = clean_text(
                "\n".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
            )
            if text:
                units.append(
                    TextUnit(
                        text=text,
                        page=slide_number,
                        section=f"Slide {slide_number}",
                        method="pptx-xml",
                    )
                )
    return ExtractedDocument(
        title=path.stem,
        units=units,
        page_count=len(units),
        parser="pptx-xml",
        diagnostics=[],
    )


def extract_document(path: Path) -> ExtractedDocument:
    extension = path.suffix.casefold()
    if extension == ".pdf":
        return extract_pdf(path)
    if extension == ".epub":
        return extract_epub(path)
    if extension in {".md", ".rst", ".txt"}:
        return extract_plain(path)
    if extension == ".doc":
        return extract_doc(path)
    if extension == ".pptx":
        return extract_pptx(path)
    raise ValueError(f"Unsupported extension: {extension}")


def cyber_category(relative_path: Path) -> str:
    if len(relative_path.parts) > 1:
        return relative_path.parts[0].upper()
    return "CYBER"


def finance_category(path: Path) -> str:
    name = path.stem.casefold()
    rules = [
        ("DERIVATIVES", r"option|future|derivative|volatility|hedg|stochastic calculus"),
        ("MARKET_MICROSTRUCTURE", r"microstructure|high.frequency|exchange|execution"),
        ("QUANT_ML", r"machine.learning|python|algorithm|reinforcement|statistics|kaggle"),
        ("PORTFOLIO_RISK", r"portfolio|asset manager|risk|allocation"),
        ("ECONOMICS_BANKING", r"economic|macro|monnaie|banque|money|central bank"),
        ("ACCOUNTING_VALUATION", r"accounting|statement|valuation|security.analysis"),
        ("TRADING", r"trading|trader|chart|candlestick|ichimoku|forex"),
        ("INVESTING", r"invest|buffett|graham|stock|market"),
    ]
    for category, pattern in rules:
        if re.search(pattern, name):
            return category
    return "FINANCE_GENERAL"


def chunk_windows(
    unit: TextUnit,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[TextUnit]:
    text = unit.text
    if len(text) <= max_chars:
        return [unit]
    output = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = max(
                text.rfind("\n\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            )
            if split_at > start + max_chars // 2:
                end = split_at + (2 if text[split_at:split_at + 2] == ". " else 0)
        part = clean_text(text[start:end])
        if part:
            output.append(
                TextUnit(
                    text=part,
                    page=unit.page,
                    section=unit.section,
                    method=unit.method,
                )
            )
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return output


def chunk_quality(text: str) -> tuple[bool, float, int]:
    compact = canonical_text(text)
    words = WORD_RE.findall(text)
    nonspace = [char for char in text if not char.isspace()]
    if not compact or len(text) < 200 or len(words) < 30:
        return False, 0.0, len(words)
    printable_ratio = sum(char.isprintable() for char in text) / max(1, len(text))
    alpha_ratio = sum(char.isalpha() for char in nonspace) / max(1, len(nonspace))
    replacement_ratio = text.count("\ufffd") / max(1, len(text))
    score = max(
        0.0,
        min(1.0, 0.55 * alpha_ratio + 0.45 * printable_ratio - 5 * replacement_ratio),
    )
    keep = printable_ratio >= 0.96 and alpha_ratio >= 0.45 and replacement_ratio <= 0.002
    return keep, score, len(words)


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA page_size=4096;
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA foreign_keys=ON;

        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            document_uid TEXT NOT NULL UNIQUE,
            domain TEXT NOT NULL,
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            source_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            extension TEXT NOT NULL,
            media_type TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            source_size INTEGER NOT NULL,
            page_count INTEGER,
            parser TEXT,
            extraction_status TEXT NOT NULL,
            extracted_char_count INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            duplicate_of_document_id INTEGER,
            diagnostics_json TEXT NOT NULL DEFAULT '[]',
            rights_status TEXT NOT NULL DEFAULT 'unknown_review_required',
            FOREIGN KEY (duplicate_of_document_id) REFERENCES documents(id)
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            chunk_uid TEXT NOT NULL UNIQUE,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            section TEXT NOT NULL DEFAULT '',
            page_start INTEGER,
            page_end INTEGER,
            text TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            token_estimate INTEGER NOT NULL,
            quality_score REAL NOT NULL,
            extraction_method TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );

        CREATE INDEX idx_documents_category ON documents(category);
        CREATE INDEX idx_documents_sha256 ON documents(source_sha256);
        CREATE INDEX idx_chunks_document ON chunks(document_id, chunk_index);
        CREATE INDEX idx_chunks_category ON chunks(category);
        CREATE INDEX idx_chunks_content_sha256 ON chunks(content_sha256);

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            title,
            section,
            category,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )


def mime_type(extension: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".epub": "application/epub+zip",
        ".md": "text/markdown",
        ".rst": "text/x-rst",
        ".txt": "text/plain",
        ".doc": "application/msword",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }.get(extension, "application/octet-stream")


def build_database(
    *,
    domain: str,
    source_root: Path,
    output_path: Path,
    max_chars: int,
    overlap_chars: int,
) -> dict:
    files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    connection = sqlite3.connect(temporary_path)
    create_schema(connection)

    stats: collections.Counter[str] = collections.Counter()
    category_counts: collections.Counter[str] = collections.Counter()
    seen_documents: dict[str, int] = {}
    seen_chunks: set[str] = set()
    start_time = time.time()

    try:
        for position, path in enumerate(files, 1):
            relative = path.relative_to(source_root)
            category = (
                cyber_category(relative)
                if domain == "cyber"
                else finance_category(path)
            )
            extension = path.suffix.casefold()
            source_hash = sha256_file(path)
            duplicate_of = seen_documents.get(source_hash)
            status = "duplicate" if duplicate_of else "pending"
            extracted = None
            error = None

            if duplicate_of is None:
                try:
                    extracted = extract_document(path)
                    status = "indexed" if extracted.units else "no_native_text"
                except Exception as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"

            cursor = connection.execute(
                """
                INSERT INTO documents (
                    document_uid, domain, category, title, source_name,
                    source_path, relative_path, extension, media_type,
                    source_sha256, source_size, page_count, parser,
                    extraction_status, duplicate_of_document_id, diagnostics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        f"sha256:{source_hash}:"
                        f"{hashlib.sha256(str(relative).encode()).hexdigest()[:12]}"
                    ),
                    domain,
                    category,
                    extracted.title if extracted else path.stem,
                    path.name,
                    str(path.resolve()),
                    str(relative),
                    extension,
                    mime_type(extension),
                    source_hash,
                    path.stat().st_size,
                    extracted.page_count if extracted else None,
                    extracted.parser if extracted else None,
                    status,
                    duplicate_of,
                    json.dumps(
                        (extracted.diagnostics if extracted else [])
                        + ([error] if error else []),
                        ensure_ascii=False,
                    ),
                ),
            )
            document_id = cursor.lastrowid
            if duplicate_of is None:
                seen_documents[source_hash] = document_id

            chunk_count = 0
            extracted_chars = 0
            if extracted:
                local_index = 0
                for unit in extracted.units:
                    extracted_chars += len(unit.text)
                    for window in chunk_windows(
                        unit,
                        max_chars=max_chars,
                        overlap_chars=overlap_chars,
                    ):
                        keep, quality, word_count = chunk_quality(window.text)
                        if not keep:
                            stats["chunks_rejected_quality"] += 1
                            continue
                        content_hash = hashlib.sha256(
                            canonical_text(window.text).encode()
                        ).hexdigest()
                        if content_hash in seen_chunks:
                            stats["chunks_rejected_duplicate"] += 1
                            continue
                        seen_chunks.add(content_hash)
                        chunk_uid = hashlib.sha256(
                            f"{domain}:{source_hash}:{local_index}:{content_hash}".encode()
                        ).hexdigest()
                        connection.execute(
                            """
                            INSERT INTO chunks (
                                chunk_uid, document_id, chunk_index, category,
                                title, section, page_start, page_end, text,
                                content_sha256, char_count, word_count,
                                token_estimate, quality_score, extraction_method
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                f"sha256:{chunk_uid}",
                                document_id,
                                local_index,
                                category,
                                extracted.title,
                                window.section,
                                window.page,
                                window.page,
                                window.text,
                                content_hash,
                                len(window.text),
                                word_count,
                                math.ceil(len(window.text) / 4),
                                round(quality, 4),
                                window.method,
                            ),
                        )
                        local_index += 1
                        chunk_count += 1

            connection.execute(
                """
                UPDATE documents
                SET extracted_char_count = ?, chunk_count = ?
                WHERE id = ?
                """,
                (extracted_chars, chunk_count, document_id),
            )
            stats[f"documents_{status}"] += 1
            stats["documents_total"] += 1
            stats["chunks_indexed"] += chunk_count
            stats["extracted_chars"] += extracted_chars
            category_counts[category] += chunk_count
            if position % 10 == 0 or position == len(files):
                elapsed = time.time() - start_time
                print(
                    f"[{domain}] {position}/{len(files)} documents | "
                    f"{stats['chunks_indexed']} chunks | {elapsed:.1f}s",
                    flush=True,
                )
            if position % 25 == 0:
                connection.commit()

        connection.execute(
            """
            INSERT INTO chunks_fts(rowid, text, title, section, category)
            SELECT id, text, title, section, category FROM chunks
            """
        )
        metadata = {
            "schema_version": "1",
            "domain": domain,
            "created_at_unix": str(int(time.time())),
            "source_root": str(source_root.resolve()),
            "chunk_max_chars": str(max_chars),
            "chunk_overlap_chars": str(overlap_chars),
            "extractor": "native-originals-no-ocr",
            "retrieval": "SQLite FTS5 / BM25",
            "rights_status": "unknown_review_required",
            "stats_json": json.dumps(dict(stats), sort_keys=True),
            "category_counts_json": json.dumps(dict(category_counts), sort_keys=True),
        }
        connection.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.execute("VACUUM")
        connection.close()
        os.replace(temporary_path, output_path)
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "domain": domain,
        "output": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "stats": dict(stats),
        "category_counts": dict(category_counts),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cyber-source",
        type=Path,
        default=ROOT / "data" / "source" / "00_DATA_SOURCE_LLM",
    )
    parser.add_argument(
        "--finance-source",
        type=Path,
        default=ROOT.parent / "DATASETS" / "FINANCE" / "FINANCE_DOCS",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "rag",
    )
    parser.add_argument("--max-chars", type=int, default=3200)
    parser.add_argument("--overlap-chars", type=int, default=320)
    parser.add_argument(
        "--domain",
        choices=("both", "cyber", "finance"),
        default="both",
    )
    args = parser.parse_args()

    reports = []
    if args.domain in {"both", "cyber"}:
        reports.append(
            build_database(
                domain="cyber",
                source_root=args.cyber_source,
                output_path=args.output_dir / "cyber_rag.sqlite",
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
            )
        )
    if args.domain in {"both", "finance"}:
        reports.append(
            build_database(
                domain="finance",
                source_root=args.finance_source,
                output_path=args.output_dir / "finance_rag.sqlite",
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
            )
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
