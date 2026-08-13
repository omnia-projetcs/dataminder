"""Export helpers for cleaned Alpaca JSONL and Hugging Face AutoTrain files."""

from __future__ import annotations

import json
import os

from processing_manifest import atomic_write_text


def cleaned_jsonl_path(input_file: str) -> str:
    if input_file.endswith(".jsonl"):
        return input_file[:-6] + "_cleaned.jsonl"
    if input_file.endswith(".json"):
        return input_file[:-5] + "_cleaned.jsonl"
    return f"{input_file}_cleaned.jsonl"


def _iter_alpaca_records(raw: str):
    stripped = raw.lstrip()
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError("expected a JSON list of Alpaca records")
        yield from data
        return
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


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
    data = None
    input_desc = ""

    if isinstance(data_or_filepath, str):
        input_file = data_or_filepath
        if not os.path.exists(input_file):
            print(f"Clean dataset skipped: file not found: {input_file}")
            return None
        input_desc = input_file
        if output_file is None:
            output_file = cleaned_jsonl_path(input_file)
        try:
            with open(input_file, "r", encoding="utf-8") as infile:
                data = json.load(infile)
        except Exception as e:
            print(f"Could not read '{input_file}': {e}")
            return None
    elif isinstance(data_or_filepath, list):
        data = data_or_filepath
        input_desc = "in-memory list"
        if output_file is None:
            output_file = "dataset_cleaned.jsonl"
    else:
        print(f"Invalid data type for clean_dataset: {type(data_or_filepath)}")
        return None

    if not isinstance(data, list):
        print(f"Unexpected format in '{input_desc}': expected a JSON list.")
        return None

    print(f"Cleaning dataset '{input_desc}' -> '{output_file}'...")

    try:
        lines = []
        for item in data:
            if isinstance(item, dict):
                instruction = str(item.get("instruction") or "").strip()
                output = str(item.get("output") or "").strip()
                input_val = str(item.get("input") or "").strip()

                if instruction and output:
                    clean_entry = {
                        "instruction": instruction,
                        "input": input_val,
                        "output": output,
                    }
                    lines.append(json.dumps(clean_entry, ensure_ascii=False))

        atomic_write_text(
            output_file,
            "\n".join(lines) + ("\n" if lines else ""),
        )
        print(f"Wrote {len(lines)} valid lines to '{output_file}'")
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
        with open(input_file, "r", encoding="utf-8") as infile:
            raw = infile.read()
        lines = []
        for data in _iter_alpaca_records(raw):
            if not isinstance(data, dict):
                continue
            inst = str(data.get("instruction") or "").strip()
            inp = str(data.get("input") or "").strip()
            out = str(data.get("output") or "").strip()
            if not inst or not out:
                continue
            lines.append(json.dumps({"text": format_hf_text(inst, inp, out)}, ensure_ascii=False))

        atomic_write_text(
            output_file,
            "\n".join(lines) + ("\n" if lines else ""),
        )
        print(f"Wrote Hugging Face file '{output_file}' ({len(lines)} examples).")
        return output_file
    except Exception as e:
        print(f"Could not prepare Hugging Face export for '{input_file}': {e}")
        return None
