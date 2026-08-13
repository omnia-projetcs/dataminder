"""Export helpers for cleaned Alpaca JSONL and Hugging Face AutoTrain files."""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

from processing_manifest import atomic_text_writer


def cleaned_jsonl_path(input_file: str) -> str:
    if input_file.endswith(".jsonl"):
        return input_file[:-6] + "_cleaned.jsonl"
    if input_file.endswith(".json"):
        return input_file[:-5] + "_cleaned.jsonl"
    return f"{input_file}_cleaned.jsonl"


def iter_alpaca_records(source) -> Iterator[dict]:
    """Yield Alpaca records from a list, JSON array file, or JSONL file."""
    if isinstance(source, list):
        for item in source:
            if isinstance(item, dict):
                yield item
        return

    with open(source, "r", encoding="utf-8-sig") as handle:
        while True:
            position = handle.tell()
            char = handle.read(1)
            if not char:
                return
            if not char.isspace():
                handle.seek(position)
                break
        if char == "[":
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError("expected a JSON list of Alpaca records")
            for item in data:
                if isinstance(item, dict):
                    yield item
            return
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def _normalized_record(item: dict) -> dict | None:
    instruction = str(item.get("instruction") or "").strip()
    output = str(item.get("output") or "").strip()
    input_val = str(item.get("input") or "").strip()
    if not instruction or not output:
        return None
    return {
        "instruction": instruction,
        "input": input_val,
        "output": output,
    }


def _write_jsonl(path: str, records: Iterable[dict]) -> int:
    count = 0
    with atomic_text_writer(path) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def format_hf_text(instruction: str, input_value: str, output: str) -> str:
    if input_value:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_value}\n\n"
            f"### Response:\n{output}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def clean_dataset(data_or_filepath, output_file=None):
    """Write a validated Alpaca JSONL file (one object per line)."""
    if isinstance(data_or_filepath, str):
        if not os.path.exists(data_or_filepath):
            print(f"Clean dataset skipped: file not found: {data_or_filepath}")
            return None
        input_desc = data_or_filepath
        source = data_or_filepath
        if output_file is None:
            output_file = cleaned_jsonl_path(data_or_filepath)
    elif isinstance(data_or_filepath, list):
        input_desc = "in-memory list"
        source = data_or_filepath
        if output_file is None:
            output_file = "dataset_cleaned.jsonl"
    else:
        print(f"Invalid data type for clean_dataset: {type(data_or_filepath)}")
        return None

    print(f"Cleaning dataset '{input_desc}' -> '{output_file}'...")

    try:
        records = (
            cleaned
            for item in iter_alpaca_records(source)
            if (cleaned := _normalized_record(item)) is not None
        )
        count = _write_jsonl(output_file, records)
        print(f"Wrote {count} valid lines to '{output_file}'")
        return output_file
    except Exception as e:
        print(f"Could not clean '{input_desc}': {e}")
        return None


def prepare_hf_dataset(input_file, output_file=None):
    """Convert cleaned Alpaca JSON/JSONL into Hugging Face `{"text": ...}` JSONL."""
    if not os.path.exists(input_file):
        print(f"Hugging Face export skipped: file not found: {input_file}")
        return None

    if output_file is None:
        if input_file.endswith(".json") or input_file.endswith(".jsonl"):
            base = os.path.splitext(input_file)[0]
            output_file = f"{base}_hf.jsonl"
        else:
            output_file = f"{input_file}_hf.jsonl"

    print(f"Preparing Hugging Face export '{input_file}' -> '{output_file}'...")

    try:
        def records():
            for item in iter_alpaca_records(input_file):
                cleaned = _normalized_record(item)
                if cleaned is None:
                    continue
                yield {
                    "text": format_hf_text(
                        cleaned["instruction"],
                        cleaned["input"],
                        cleaned["output"],
                    )
                }

        count = _write_jsonl(output_file, records())
        print(f"Wrote Hugging Face file '{output_file}' ({count} examples).")
        return output_file
    except Exception as e:
        print(f"Could not prepare Hugging Face export for '{input_file}': {e}")
        return None
