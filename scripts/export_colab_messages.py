#!/usr/bin/env python3
"""Convert documentary enrichment JSONL into grounded chat-training records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = (
    "Continue fidèlement le passage documentaire ci-dessous. "
    "N'ajoute aucune information absente de la source."
)


def split_documentary_text(text: str) -> tuple[str, str]:
    """Split near 30%, preferably on a paragraph or sentence boundary."""
    text = text.strip()
    minimum_context = 140
    minimum_answer = 220
    lower = minimum_context
    upper = len(text) - minimum_answer
    if upper <= lower:
        raise ValueError("Text is too short for a grounded continuation example")

    target = min(upper, max(lower, round(len(text) * 0.30)))
    candidates: set[int] = set()
    for marker in ("\n\n", ". ", "? ", "! ", ": "):
        start = max(lower, target - 240)
        end = min(upper, target + 240)
        position = text.find(marker, start, end)
        while position >= 0:
            candidates.add(position + len(marker))
            position = text.find(marker, position + len(marker), end)
    split_at = min(candidates, key=lambda value: abs(value - target)) if candidates else target
    return text[:split_at].strip(), text[split_at:].strip()


def convert_file(source: Path, output: Path, expected_domain: str) -> dict:
    if not source.is_file():
        raise FileNotFoundError(
            f"model-enrichment JSONL not found: {source}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    count = 0
    input_characters = 0
    answer_characters = 0
    seen_messages: set[str] = set()

    try:
        with (
            source.open("r", encoding="utf-8-sig") as source_handle,
            os.fdopen(descriptor, "w", encoding="utf-8") as output_handle,
        ):
            for line_number, line in enumerate(source_handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{source}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if record.get("domain") != expected_domain:
                    raise ValueError(
                        f"{source}:{line_number}: unexpected domain "
                        f"{record.get('domain')!r}"
                    )
                context, answer = split_documentary_text(record["text"])
                user_content = f"{INSTRUCTION}\n\n{context}"
                message_key = hashlib.sha256(
                    f"{user_content}\n{answer}".encode("utf-8")
                ).hexdigest()
                if message_key in seen_messages:
                    raise ValueError(f"{source}:{line_number}: duplicate messages")
                seen_messages.add(message_key)

                converted = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {
                        "domain": record["domain"],
                        "category": record["category"],
                        "source_title": record["source_title"],
                        "section": record["section"],
                        "page_start": record["page_start"],
                        "page_end": record["page_end"],
                        "source_sha256": record["source_sha256"],
                        "rights_status": record["rights_status"],
                        "chunk_uid": record["chunk_uid"],
                        "quality_score": record["quality_score"],
                        "task": "grounded_document_continuation",
                    },
                }
                output_handle.write(
                    json.dumps(
                        converted,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                count += 1
                input_characters += len(context)
                answer_characters += len(answer)

            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    return {
        "domain": expected_domain,
        "source": str(source),
        "output": str(output),
        "records": count,
        "size_bytes": output.stat().st_size,
        "context_characters": input_characters,
        "answer_characters": answer_characters,
        "task": "grounded_document_continuation",
        "format": "messages[user, assistant]",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data" / "model_enrichment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "model_enrichment_colab",
    )
    args = parser.parse_args()

    reports = []
    for domain in ("cyber", "finance"):
        reports.append(
            convert_file(
                source=args.source_dir / f"{domain}_model_enrichment.jsonl",
                output=args.output_dir / f"{domain}_colab_messages.jsonl",
                expected_domain=domain,
            )
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
