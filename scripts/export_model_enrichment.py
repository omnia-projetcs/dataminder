#!/usr/bin/env python3
"""Export conservative continued-pretraining JSONL files from the RAG databases."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"\b[\wÀ-ÿ'-]+\b", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
DOT_LEADER_RE = re.compile(r"(?:\.\s*){8,}")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
FRONT_MATTER_RE = re.compile(
    r"other titles in the|library of congress catalog(?:ing)?|"
    r"all rights reserved|isbn(?:-1[03])?",
    re.IGNORECASE,
)
COPYRIGHT_NOISE_RE = re.compile(
    r"copyright|reproduction|all rights reserved",
    re.IGNORECASE,
)
LICENSE_FRONT_MATTER_RE = re.compile(
    r"courseware license agreement|important\s*[-–—]?\s*read carefully|"
    r"complete and exclusive statement of agreement",
    re.IGNORECASE,
)
INDEX_REFERENCE_RE = re.compile(r",\s*\d+(?:\s*[-–]\s*\d+)?")
CYBER_OFF_DOMAIN_TITLES = {
    "learning python 5e",
    "python for data analysis",
    "introduction to machine learning with python",
    "b072ptrzsj ebok",
}


def canonical_text(value: str) -> str:
    return SPACE_RE.sub(" ", value).strip().casefold()


def evenly_spaced_indices(count: int, limit: int) -> set[int]:
    """Keep full-document coverage when a large source must be capped."""
    if count <= limit:
        return set(range(count))
    if limit == 1:
        return {count // 2}
    return {
        round(position * (count - 1) / (limit - 1))
        for position in range(limit)
    }


def trim_exact_overlap(previous: str, current: str) -> tuple[str, int]:
    """Remove chunk overlap without fuzzy rewriting or loss of provenance."""
    minimum = 80
    maximum = min(600, len(previous), len(current))
    if maximum < minimum:
        return current, 0
    anchor = current[:minimum]
    position = previous.rfind(anchor, max(0, len(previous) - maximum))
    if position < 0:
        return current, 0
    overlap = previous[position:]
    if current.startswith(overlap):
        return current[len(overlap):].lstrip(), len(overlap)
    return current, 0


def document_rejection_reason(domain: str, title: str) -> str | None:
    normalized = title.casefold()
    if domain == "cyber":
        if (
            "powerpoint" in normalized
            or ".ppt" in normalized
            or "présentation powerpoint" in normalized
        ):
            return "presentation_layout"
        if any(marker in normalized for marker in ("catalog", "poster")):
            return "catalog_or_poster"
        if any(marker in normalized for marker in ("lab setup", "quick find chart")):
            return "lab_setup_or_reference_card"
        if normalized in CYBER_OFF_DOMAIN_TITLES:
            return "off_domain"
    if domain == "finance" and "creature from jekyll island" in normalized:
        return "untrusted_source"
    return None


def training_text_rejection_reason(
    text: str,
    *,
    min_chars: int,
    min_words: int,
) -> str | None:
    words = WORD_RE.findall(text)
    if len(text) < min_chars or len(words) < min_words:
        return "too_short"
    if text.count("\ufffd") / max(1, len(text)) > 0.001:
        return "replacement_characters"
    if len(DOT_LEADER_RE.findall(text)) >= 4:
        return "dot_leaders"
    if len(URL_RE.findall(text)) >= 12:
        return "url_list"
    if (
        FRONT_MATTER_RE.search(text)
        and (
            len(FRONT_MATTER_RE.findall(text)) >= 2
            or re.search(
                r"other titles in the|library of congress catalog(?:ing)?",
                text,
                re.IGNORECASE,
            )
        )
    ):
        return "publisher_front_matter"
    if len(COPYRIGHT_NOISE_RE.findall(text)) >= 3:
        return "copyright_noise"
    if LICENSE_FRONT_MATTER_RE.search(text):
        return "license_front_matter"
    if (
        len(INDEX_REFERENCE_RE.findall(text)) >= 12
        and text.count(".") <= 3
    ):
        return "index_or_reference_list"
    unique_ratio = len({word.casefold() for word in words}) / len(words)
    if unique_ratio < 0.12:
        return "low_vocabulary_diversity"
    return None


def database_domain(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT value FROM meta WHERE key = 'domain'"
    ).fetchone()
    if row is None or row[0] not in {"cyber", "finance"}:
        raise ValueError("RAG database has no valid domain metadata")
    return row[0]


def export_database(
    *,
    database: Path,
    output: Path,
    expected_domain: str,
    min_quality: float,
    min_chars: int,
    min_words: int,
    max_chunks_per_document: int,
) -> dict:
    if not database.is_file():
        raise FileNotFoundError(f"RAG database not found: {database}")
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    domain = database_domain(connection)
    if domain != expected_domain:
        raise ValueError(
            f"{database} contains domain {domain!r}, expected {expected_domain!r}"
        )

    documents = connection.execute(
        """
        SELECT id, document_uid, title, category, source_sha256, chunk_count,
               rights_status
        FROM documents
        WHERE extraction_status = 'indexed'
        ORDER BY id
        """
    ).fetchall()

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    counters: collections.Counter[str] = collections.Counter()
    category_counts: collections.Counter[str] = collections.Counter()
    seen_text: set[str] = set()
    source_counts: dict[str, int] = {}
    title_counts: collections.Counter[str] = collections.Counter()

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for document in documents:
                chunks = connection.execute(
                    """
                    SELECT chunk_uid, chunk_index, section, page_start, page_end,
                           text, quality_score, extraction_method
                    FROM chunks
                    WHERE document_id = ? AND quality_score >= ?
                    ORDER BY chunk_index
                    """,
                    (document["id"], min_quality),
                ).fetchall()
                document_reason = document_rejection_reason(
                    domain, document["title"]
                )
                if document_reason:
                    counters[f"rejected_{document_reason}"] += len(chunks)
                    continue
                selected_positions = evenly_spaced_indices(
                    len(chunks), max_chunks_per_document
                )
                if len(chunks) > max_chunks_per_document:
                    counters["rejected_document_cap"] += (
                        len(chunks) - len(selected_positions)
                    )

                previous_text = ""
                previous_index: int | None = None
                written_for_source = 0
                for position, chunk in enumerate(chunks):
                    if position not in selected_positions:
                        continue
                    text = chunk["text"].strip()
                    overlap_removed = 0
                    if (
                        previous_index is not None
                        and chunk["chunk_index"] == previous_index + 1
                    ):
                        text, overlap_removed = trim_exact_overlap(previous_text, text)
                    previous_text = chunk["text"]
                    previous_index = chunk["chunk_index"]

                    rejection_reason = training_text_rejection_reason(
                        text, min_chars=min_chars, min_words=min_words
                    )
                    if rejection_reason:
                        counters[f"rejected_{rejection_reason}"] += 1
                        continue

                    content_key = hashlib.sha256(
                        canonical_text(text).encode("utf-8")
                    ).hexdigest()
                    if content_key in seen_text:
                        counters["rejected_duplicate"] += 1
                        continue
                    title_key = canonical_text(document["title"])
                    if title_counts[title_key] >= max_chunks_per_document:
                        counters["rejected_title_cap"] += 1
                        continue
                    seen_text.add(content_key)

                    record = {
                        "text": text,
                        "domain": domain,
                        "category": document["category"],
                        "source_title": document["title"],
                        "section": chunk["section"],
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                        "source_sha256": document["source_sha256"],
                        "rights_status": document["rights_status"],
                        "chunk_uid": chunk["chunk_uid"],
                        "quality_score": chunk["quality_score"],
                        "extraction_method": chunk["extraction_method"],
                    }
                    handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    counters["records_written"] += 1
                    counters["characters_written"] += len(text)
                    counters["overlap_characters_removed"] += overlap_removed
                    category_counts[document["category"]] += 1
                    title_counts[title_key] += 1
                    written_for_source += 1

                if written_for_source:
                    source_counts[document["document_uid"]] = written_for_source

            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    finally:
        connection.close()

    return {
        "domain": domain,
        "output": str(output),
        "size_bytes": output.stat().st_size,
        "records": counters["records_written"],
        "characters": counters["characters_written"],
        "sources_represented": len(source_counts),
        "largest_source_records": max(source_counts.values(), default=0),
        "largest_title_records": max(title_counts.values(), default=0),
        "category_counts": dict(sorted(category_counts.items())),
        "filters": {
            "minimum_rag_quality": min_quality,
            "minimum_characters": min_chars,
            "minimum_words": min_words,
            "maximum_chunks_per_document": max_chunks_per_document,
        },
        "rejections": {
            key.removeprefix("rejected_"): value
            for key, value in sorted(counters.items())
            if key.startswith("rejected_")
        },
        "overlap_characters_removed": counters["overlap_characters_removed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rag-dir",
        type=Path,
        default=ROOT / "data" / "rag",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "model_enrichment",
    )
    parser.add_argument("--min-quality", type=float, default=0.80)
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--max-chunks-per-document", type=int, default=600)
    args = parser.parse_args()

    reports = []
    for domain in ("cyber", "finance"):
        reports.append(
            export_database(
                database=args.rag_dir / f"{domain}_rag.sqlite",
                output=args.output_dir / f"{domain}_model_enrichment.jsonl",
                expected_domain=domain,
                min_quality=args.min_quality,
                min_chars=args.min_chars,
                min_words=args.min_words,
                max_chunks_per_document=args.max_chunks_per_document,
            )
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
