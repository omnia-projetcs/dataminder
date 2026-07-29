#!/usr/bin/env python3
"""Build conservative, traceable cleaned copies of instruction JSONL corpora."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import tempfile
from pathlib import Path
from typing import TextIO


SECTION_RE = re.compile(
    r"\A### Instruction:\n(?P<instruction>.*?)"
    r"(?:\n\n### Input:\n(?P<input>.*?))?"
    r"\n\n### Response:\n(?P<response>.*)\Z",
    re.DOTALL,
)
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)

CONTEXT_DEPENDENT_RE = re.compile(
    r"\b(?:"
    r"according to (?:chapter|figure|table|the (?:analysis|author|book|document|"
    r"example|provided (?:context|information|text)|text))|"
    r"(?:figure|table|chapter)\s+\d+(?:\.\d+)*|"
    r"the author(?:'s|s')?\b|"
    r"this (?:book|chapter|document|example|section|text)\b|"
    r"(?:shown|used|described|presented) in the (?:example|figure|table)\b|"
    r"the (?:above|below|following|preceding) (?:example|figure|table|text)\b|"
    r"at the time of writing"
    r")",
    re.IGNORECASE,
)
UNANCHORED_CURRENT_RE = re.compile(
    r"\b(?:"
    r"what (?:is|are|was|were) the current\b|"
    r"current(?:ly)? (?:recommended|supported|available|trading|priced|worth)|"
    r"current (?:share |stock |market )?(?:price|value|level|rate|spread|volume|version|status)|"
    r"latest (?:stable )?(?:version|release|framework)|"
    r"where can the latest version"
    r")",
    re.IGNORECASE,
)
DATE_ANCHOR_RE = re.compile(
    r"\b(?:as of|in|on|during|for)\s+(?:"
    r"(?:19|20)\d{2}|"
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b|"
    r"android\s+\d"
    r")",
    re.IGNORECASE,
)
NONANSWER_RE = re.compile(
    r"^(?:"
    r"n/?a(?:\s|$)|"
    r"unknown(?:\s|\(|$)|"
    r"not (?:specified|provided|given|available|mentioned)(?:\s|,|\.|$)|"
    r"(?:the |an? )?(?:answer|cause|date|definition|detail|formula|information|"
    r"percentage|purpose|rate|reason|value|year).{0,80}\b(?:is|are|was|were) "
    r"not (?:specified|provided|given|available|mentioned)\b|"
    r"cannot be determined\b|"
    r"the exact formula is not provided\b"
    r")",
    re.IGNORECASE,
)
QUESTION_REQUEST_RE = re.compile(
    r"\b(?:what (?:question|questions)|give an example of a question|"
    r"which question|key questions?|questions? (?:should|did|does|can))\b",
    re.IGNORECASE,
)

FINANCE_RE = re.compile(
    r"\b(?:"
    r"account(?:ing|s?)|acquisition|alpha|amorti[sz]ation|annuit(?:y|ies)|arbitrage|"
    r"asset|auction|balance sheet|bank(?:ing|s?)?|basis points?|bear market|beta|"
    r"bid|bond|book value|broker(?:age|s?)?|budget|bull market|capital|cash(?: flow)?|"
    r"candlestick|collateral|commodit(?:y|ies)|commission|corporat(?:e|ion)|"
    r"correlation|covariance|credit|currenc(?:y|ies)|debt|derivative|discount|"
    r"dividend|dow jones|earnings?|econom(?:ic|ics|y|etric)|equity|exchange rate|"
    r"fasb|financ(?:e|ial|ing)|fiscal|forex|fund|futures?|gdp|goodwill|"
    r"hedg(?:e|ing)|inflation|insur(?:ance|er)|interest|invest(?:ment|ors?|ing)|"
    r"liquidit(?:y|ies)|loan|macroeconomic|margin|market|monetary|mortgage|option|"
    r"p/[be]\b|portfolio|pric(?:e|ing)|profit|loss|quant|recession|return|risk|"
    r"s&p(?:\s*500)?|securit(?:y|ies)|shareholders?|shares?|slippage|"
    r"stock(?:broker|holder|s)?|swap|tasuki|tax|technical analysis|"
    r"transaction costs?|trade|traders?|trading|treasury|valuation|variance|"
    r"volatility|wealth|yield|[$€£¥]\s?\d"
    r")\b",
    re.IGNORECASE,
)
CYBER_RE = re.compile(
    r"\b(?:"
    r"access|account|active directory|adb|administrator|aircrack|amap|android|"
    r"array|attack|audit|authenticat|authori[sz]|aws|bash|bluetooth|boot ?loader|"
    r"browser|buffer|certificate|cipher|class|cloud|cluster|code|command|computer|"
    r"container|credential|crypt|cve|database|dataframe|debug|device|dictionary|"
    r"directory|disaster recovery|dns|docker|encrypt|exception|exploit|feature|"
    r"file|firewall|forensic|function|hash|https?|impersonat|incident|injection|"
    r"ip address|iptables|java|kernel|key ?store|least privilege|linux|log(?:ging)?|"
    r"machine learning|malware|memory|metasploit|method|mimikatz|model|msf(?:cli|console)?|"
    r"nat|network|nmap|numpy|oracle|overflow|packet|pandas|password|permission|"
    r"policy|process|programming|protocol|python|rdp|registry|regression|"
    r"reverse engineering|root|s3|sandbox|server|shell|siem|sklearn|smtp|"
    r"software|sql|ssh|system|tcp|threat|tls|token|unix|user|variable|"
    r"vulnerabilit|web|windows|wireless"
    r")",
    re.IGNORECASE,
)

KNOWN_BAD_INSTRUCTIONS = {
    "how does lightgbm perform with categorical encoding compared to one hot encoding?",
    "how do utf-16 and utf-32 encodings differ from other schemes?",
    "what is the role of the black-schwartz option pricing model?",
}


def normalize(value: str) -> str:
    return SPACE_RE.sub(" ", value).strip().casefold()


def normalize_alnum(value: str) -> str:
    return " ".join(WORD_RE.findall(value.casefold()))


def infer_domain(path: Path) -> str:
    name = path.name.casefold()
    if "finance" in name:
        return "finance"
    if "cyber" in name:
        return "cyber"
    return "unknown"


def rejection_reasons(
    instruction: str,
    input_text: str | None,
    response: str,
    domain: str,
    seen_instructions: set[str],
) -> list[str]:
    reasons: list[str] = []
    normalized_instruction = normalize(instruction)
    normalized_response = normalize(response)
    dedupe_key = normalize_alnum(instruction)

    if input_text is not None:
        reasons.append("synthetic_input")
    if not instruction:
        reasons.append("empty_instruction")
    if not response:
        reasons.append("empty_response")
    if "\ufffd" in instruction or "\ufffd" in response:
        reasons.append("corrupted_unicode")
    if normalized_instruction in KNOWN_BAD_INSTRUCTIONS:
        reasons.append("known_bad_answer")
    if NONANSWER_RE.search(response):
        reasons.append("explicit_nonanswer")
    if CONTEXT_DEPENDENT_RE.search(instruction):
        reasons.append("missing_source_context")
    if UNANCHORED_CURRENT_RE.search(instruction) and not DATE_ANCHOR_RE.search(instruction):
        reasons.append("unanchored_time_sensitive")
    if normalized_instruction == normalized_response:
        reasons.append("answer_echoes_question")
    if (
        response.endswith("?")
        and not QUESTION_REQUEST_RE.search(instruction)
        and normalized_instruction != normalized_response
    ):
        q_words = set(WORD_RE.findall(instruction.casefold()))
        a_words = set(WORD_RE.findall(response.casefold()))
        overlap = len(q_words & a_words) / max(1, min(len(q_words), len(a_words)))
        if overlap < 0.35:
            reasons.append("likely_mismatched_answer")
    if response.count("`") % 2:
        reasons.append("unbalanced_backticks")
    if dedupe_key in seen_instructions:
        reasons.append("duplicate_instruction")

    combined = f"{instruction}\n{response}"
    if domain == "finance" and not FINANCE_RE.search(combined):
        reasons.append("off_domain")
    elif domain == "cyber" and not CYBER_RE.search(combined):
        reasons.append("off_domain")

    return reasons


def output_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    stem = input_path.name.removesuffix(".jsonl")
    return (
        output_dir / f"{stem}.clean.jsonl",
        output_dir / f"{stem}.rejected.jsonl",
        output_dir / f"{stem}.cleaning-report.json",
    )


def open_atomic(target: Path) -> tuple[TextIO, Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )
    return os.fdopen(descriptor, "w", encoding="utf-8"), Path(temporary)


def clean_file(input_path: Path, output_dir: Path, dry_run: bool) -> dict:
    domain = infer_domain(input_path)
    clean_path, rejected_path, report_path = output_paths(input_path, output_dir)
    counts: collections.Counter[str] = collections.Counter()
    seen_instructions: set[str] = set()
    clean_handle = rejected_handle = None
    clean_tmp = rejected_tmp = None

    if not dry_run:
        clean_handle, clean_tmp = open_atomic(clean_path)
        rejected_handle, rejected_tmp = open_atomic(rejected_path)

    try:
        with input_path.open("r", encoding="utf-8", errors="strict") as source:
            for line_number, raw_line in enumerate(source, 1):
                counts["total"] += 1
                reasons: list[str] = []
                instruction = ""
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError:
                    reasons.append("invalid_json")
                    value = None

                if not isinstance(value, dict) or set(value) != {"text"}:
                    reasons.append("invalid_schema")
                    text = ""
                else:
                    text = value["text"]
                    if not isinstance(text, str):
                        reasons.append("non_string_text")
                        text = ""

                match = SECTION_RE.match(text) if text else None
                if not match:
                    reasons.append("invalid_template")
                else:
                    instruction = match.group("instruction").strip()
                    input_text = match.group("input")
                    response = match.group("response").strip()
                    reasons.extend(
                        rejection_reasons(
                            instruction,
                            input_text,
                            response,
                            domain,
                            seen_instructions,
                        )
                    )

                reasons = list(dict.fromkeys(reasons))
                if reasons:
                    counts["rejected"] += 1
                    for reason in reasons:
                        counts[f"reason:{reason}"] += 1
                    if rejected_handle:
                        record = {
                            "line": line_number,
                            "reasons": reasons,
                            "instruction": instruction,
                        }
                        rejected_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    counts["kept"] += 1
                    seen_instructions.add(normalize_alnum(instruction))
                    if clean_handle:
                        clean_handle.write(raw_line)
    except Exception:
        for handle in (clean_handle, rejected_handle):
            if handle and not handle.closed:
                handle.close()
        for temporary in (clean_tmp, rejected_tmp):
            if temporary and temporary.exists():
                temporary.unlink()
        raise
    else:
        if clean_handle:
            clean_handle.flush()
            os.fsync(clean_handle.fileno())
            clean_handle.close()
        if rejected_handle:
            rejected_handle.flush()
            os.fsync(rejected_handle.fileno())
            rejected_handle.close()
        if not dry_run:
            os.replace(clean_tmp, clean_path)
            os.replace(rejected_tmp, rejected_path)

    result = {
        "input": str(input_path),
        "domain": domain,
        "clean_output": str(clean_path),
        "rejection_log": str(rejected_path),
        "counts": dict(counts),
        "retention_percent": round(100 * counts["kept"] / max(1, counts["total"]), 2),
    }
    if not dry_run:
        report_handle, report_tmp = open_atomic(report_path)
        with report_handle:
            json.dump(result, report_handle, ensure_ascii=False, indent=2)
            report_handle.write("\n")
            report_handle.flush()
            os.fsync(report_handle.fileno())
        os.replace(report_tmp, report_path)
        result["report"] = str(report_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = []
    for input_path in args.inputs:
        output_dir = args.output_dir or input_path.parent
        results.append(clean_file(input_path, output_dir, args.dry_run))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
