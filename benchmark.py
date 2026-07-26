"""Reproducible extraction benchmark for Dataminder parser backends.

Corpus manifests are JSONL files. Each case points to a source document and can
define required/forbidden phrases, expected block types/pages, a minimum text
length, and an optional reference transcription.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

from document_ir import build_chunks, sha256_file
from document_parser import PARSER_CHOICES, extract_document
from processing_manifest import atomic_write_text


@dataclass
class BenchmarkCase:
    id: str
    path: str
    required_phrases: list[str]
    forbidden_phrases: list[str]
    min_chars: int | None
    expected_pages: list[int]
    expected_block_types: list[str]
    reference_text: str | None


def _resolve_path(value: str | None, base_dir: str) -> str | None:
    if value is None or os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def load_corpus(manifest_path: str) -> list[BenchmarkCase]:
    """Load and validate a benchmark JSONL manifest."""
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    cases = []
    seen_ids = set()
    with open(manifest_path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{manifest_path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            case_id = str(item.get("id") or f"case-{line_number}")
            if case_id in seen_ids:
                raise ValueError(f"Duplicate benchmark case id: {case_id}")
            seen_ids.add(case_id)
            path = item.get("path")
            if not path:
                raise ValueError(f"{manifest_path}:{line_number}: missing 'path'")
            min_chars = item.get("min_chars")
            if min_chars is not None and int(min_chars) < 0:
                raise ValueError(f"{case_id}: min_chars must be non-negative")
            cases.append(
                BenchmarkCase(
                    id=case_id,
                    path=_resolve_path(str(path), base_dir),
                    required_phrases=[
                        str(value) for value in item.get("required_phrases", [])
                    ],
                    forbidden_phrases=[
                        str(value) for value in item.get("forbidden_phrases", [])
                    ],
                    min_chars=int(min_chars) if min_chars is not None else None,
                    expected_pages=[
                        int(value) for value in item.get("expected_pages", [])
                    ],
                    expected_block_types=[
                        str(value).lower()
                        for value in item.get("expected_block_types", [])
                    ],
                    reference_text=_resolve_path(item.get("reference_text"), base_dir),
                )
            )
    if not cases:
        raise ValueError(f"No benchmark cases found in {manifest_path}")
    return cases


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _fraction_found(expected: list[Any], actual: set[Any]) -> float:
    if not expected:
        return 1.0
    return sum(value in actual for value in expected) / len(expected)


def score_document(document, case: BenchmarkCase) -> tuple[float, dict[str, float]]:
    """Score one extracted document, averaging only configured expectations."""
    text = document.text
    normalized = _normalized_text(text)
    components = {"non_empty": 1.0 if normalized else 0.0}

    if case.required_phrases:
        required = {
            phrase: _normalized_text(phrase) in normalized
            for phrase in case.required_phrases
        }
        components["required_phrase_recall"] = sum(required.values()) / len(required)

    if case.forbidden_phrases:
        forbidden_matches = sum(
            _normalized_text(phrase) in normalized
            for phrase in case.forbidden_phrases
        )
        components["forbidden_phrase_score"] = (
            1.0 - forbidden_matches / len(case.forbidden_phrases)
        )

    if case.min_chars is not None:
        components["minimum_length"] = (
            1.0 if len(text.strip()) >= case.min_chars else 0.0
        )

    if case.expected_pages:
        actual_pages = {
            block.page for block in document.blocks if block.page is not None
        }
        components["page_recall"] = _fraction_found(
            case.expected_pages, actual_pages
        )

    if case.expected_block_types:
        actual_types = {block.block_type.lower() for block in document.blocks}
        components["block_type_recall"] = _fraction_found(
            case.expected_block_types, actual_types
        )

    if case.reference_text:
        with open(case.reference_text, "r", encoding="utf-8") as source:
            reference = _normalized_text(source.read())
        components["reference_similarity"] = difflib.SequenceMatcher(
            None, reference, normalized
        ).ratio()

    return statistics.fmean(components.values()), components


def benchmark_corpus(
    cases: list[BenchmarkCase],
    parsers: list[str],
    *,
    marker_mode: str = "fast",
    structured: bool = False,
    extractor: Callable[..., Any] = extract_document,
) -> dict[str, Any]:
    """Run every case against every parser and return a JSON-safe report."""
    report = {
        "schema_version": 1,
        "configuration": {
            "parsers": parsers,
            "marker_mode": marker_mode,
            "structured": structured,
        },
        "results": [],
        "summary": {},
    }

    for parser_name in parsers:
        for case in cases:
            started = time.perf_counter()
            result = {
                "case_id": case.id,
                "parser": parser_name,
                "path": case.path,
            }
            try:
                source_sha256 = sha256_file(case.path)
                document = extractor(
                    case.path,
                    parser=parser_name,
                    marker_mode=marker_mode,
                    structured=structured,
                    source_sha256=source_sha256,
                )
                score, components = score_document(document, case)
                result.update(
                    {
                        "status": "success" if document.text.strip() else "empty",
                        "score": round(score, 6),
                        "components": {
                            key: round(value, 6)
                            for key, value in components.items()
                        },
                        "document_id": document.id,
                        "char_count": len(document.text),
                        "block_count": len(document.blocks),
                        "chunk_count": len(build_chunks(document)),
                        "diagnostics": document.diagnostics,
                    }
                )
            except Exception as exc:
                result.update(
                    {
                        "status": "error",
                        "score": 0.0,
                        "error": str(exc),
                    }
                )
            result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            report["results"].append(result)

    for parser_name in parsers:
        parser_results = [
            result
            for result in report["results"]
            if result["parser"] == parser_name
        ]
        report["summary"][parser_name] = {
            "document_count": len(parser_results),
            "successful_documents": sum(
                result["status"] == "success" for result in parser_results
            ),
            "mean_score": round(
                statistics.fmean(result["score"] for result in parser_results),
                6,
            ),
            "mean_elapsed_seconds": round(
                statistics.fmean(
                    result["elapsed_seconds"] for result in parser_results
                ),
                6,
            ),
        }
    return report


def _parse_parsers(value: str) -> list[str]:
    parsers = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    invalid = [item for item in parsers if item not in PARSER_CHOICES]
    if not parsers or invalid:
        raise argparse.ArgumentTypeError(
            f"parsers must be comma-separated values from {PARSER_CHOICES}"
        )
    return parsers


def _print_summary(report: dict[str, Any]) -> None:
    print("Parser benchmark summary")
    print("parser\tdocuments\tsuccess\tmean_score\tmean_seconds")
    for parser_name, summary in report["summary"].items():
        print(
            f"{parser_name}\t{summary['document_count']}\t"
            f"{summary['successful_documents']}\t{summary['mean_score']:.3f}\t"
            f"{summary['mean_elapsed_seconds']:.3f}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Dataminder document parser quality."
    )
    parser.add_argument("--corpus", required=True, help="JSONL benchmark manifest.")
    parser.add_argument(
        "--parsers",
        type=_parse_parsers,
        default=["native"],
        help="Comma-separated parser backends (default: native).",
    )
    parser.add_argument(
        "--marker-mode",
        choices=["fast", "balanced"],
        default="fast",
    )
    parser.add_argument("--structured", action="store_true")
    parser.add_argument(
        "--output",
        default="benchmark-report.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Fail when any parser mean score falls below this threshold.",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score must be between 0 and 1")

    cases = load_corpus(args.corpus)
    report = benchmark_corpus(
        cases,
        args.parsers,
        marker_mode=args.marker_mode,
        structured=args.structured,
    )
    atomic_write_text(
        args.output,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _print_summary(report)
    failed = [
        parser_name
        for parser_name, summary in report["summary"].items()
        if summary["mean_score"] < args.min_score
    ]
    if failed:
        print(
            f"Quality gate failed ({args.min_score:.3f}): {', '.join(failed)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
