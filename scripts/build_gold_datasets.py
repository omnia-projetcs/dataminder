#!/usr/bin/env python3
"""Build compact, diverse training corpora from locally available QA data."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does", "for",
    "from", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "what", "when", "where", "which", "who", "why", "with",
}
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
CONCEPTUAL_QUESTION_RE = re.compile(
    r"^(?:"
    r"how|why|explain|describe|define|compare|"
    r"what (?:are|does|do|is|makes|role|purpose|causes|happens|effect|impact|"
    r"relationship|difference|advantages|benefits|risks|formula|factors|steps)|"
    r"under what|which (?:factors|method|model|strategy|conditions)"
    r")\b",
    re.IGNORECASE,
)
TRIVIA_RE = re.compile(
    r"^(?:who|when|where|in what year|what year|what date|what is the title|"
    r"what was the name|which publication|which journal)\b",
    re.IGNORECASE,
)
META_RE = re.compile(
    r"\b(?:as an ai|language model|provided (?:context|document|information|text)|"
    r"the answer is not|cannot be determined|not enough information)\b",
    re.IGNORECASE,
)
FORMULA_HEAVY_RE = re.compile(
    r"\b(?:calculate|calculation|equation|formula|mathematical expression)\b|"
    r"[=∑√∫^]|[a-z]\d\s*[()]|<su[bp]>|[ρσμβΔ]",
    re.IGNORECASE,
)
PRECISE_CONTEXTUAL_FACT_RE = re.compile(
    r"\b(?:"
    r"(?:19|20)\d{2}|"
    r"reported (?:earnings|eps|price|return|value)|"
    r"approximate (?:price|percentage|rate|return|value|yield)|"
    r"what (?:is|was) the (?:exact |current )?(?:price|percentage|rate|value|yield)|"
    r"in (?:this|the) (?:context|example|stage)|"
    r"stage \d+"
    r")\b",
    re.IGNORECASE,
)

CYBER_CATEGORIES = {
    "identity_access": re.compile(
        r"\b(?:identity|access|account|authentication|authorization|credential|"
        r"password|privilege|mfa|multi-factor|iam|user)\b", re.I
    ),
    "network_security": re.compile(
        r"\b(?:network|firewall|dns|tcp|udp|ip address|vpn|wireless|routing|"
        r"segmentation|proxy|gateway)\b", re.I
    ),
    "cloud_security": re.compile(
        r"\b(?:aws|amazon web services|azure|cloud|s3|ec2|storage|bucket|"
        r"cloudtrail|serverless)\b", re.I
    ),
    "application_security": re.compile(
        r"\b(?:application|web|owasp|injection|api|code|devsecops|software|"
        r"vulnerability|secure development)\b", re.I
    ),
    "incident_forensics": re.compile(
        r"\b(?:incident|forensic|evidence|log|monitor|detect|response|malware|"
        r"threat|event|alert|recovery)\b", re.I
    ),
    "governance_risk": re.compile(
        r"\b(?:policy|risk|control|audit|compliance|governance|standard|"
        r"framework|management system|business continuity)\b", re.I
    ),
    "cryptography": re.compile(
        r"\b(?:encrypt|cryptograph|certificate|pki|cipher|hash|key management|"
        r"signature|tls|secret)\b", re.I
    ),
    "endpoint_hardening": re.compile(
        r"\b(?:endpoint|operating system|windows|linux|device|hardening|patch|"
        r"configuration|server|workstation|mobile)\b", re.I
    ),
}
FINANCE_CATEGORIES = {
    "portfolio_risk": re.compile(
        r"\b(?:portfolio|diversification|risk|beta|alpha|asset allocation|"
        r"efficient frontier|sharpe|drawdown|value at risk|var\b|covariance)\b", re.I
    ),
    "derivatives": re.compile(
        r"\b(?:option|futures?|forward|swap|derivative|black.scholes|binomial|"
        r"strike|call|put|delta|gamma|vega|theta|implied volatility|payoff)\b", re.I
    ),
    "markets_trading": re.compile(
        r"\b(?:market|trading|trader|trade|order|liquidity|bid|ask|spread|"
        r"execution|broker|exchange|technical indicator|momentum|trend)\b", re.I
    ),
    "quantitative": re.compile(
        r"\b(?:regression|correlation|variance|probability|distribution|"
        r"time series|monte carlo|stochastic|optimization|machine learning|"
        r"algorithm|model|python|pandas|numpy|forecast|mean reversion)\b", re.I
    ),
    "economics": re.compile(
        r"\b(?:economic|economy|inflation|gdp|monetary|fiscal|interest rate|"
        r"recession|macroeconomic|microeconomic|supply|demand|central bank)\b", re.I
    ),
    "corporate_accounting": re.compile(
        r"\b(?:accounting|balance sheet|cash flow|earnings|revenue|profit|"
        r"valuation|company|corporate|dividend|equity|shareholder|capital)\b", re.I
    ),
    "banking_credit": re.compile(
        r"\b(?:bank|credit|loan|debt|bond|yield|default|collateral|mortgage|"
        r"counterparty|treasury|fixed income)\b", re.I
    ),
    "financial_planning": re.compile(
        r"\b(?:investment|investor|wealth|saving|retirement|insurance|tax|"
        r"budget|annuity|fund|financial planning)\b", re.I
    ),
}
SOURCE_FAMILY_LIMITS = {
    "CIS_AWS_STORAGE": 0,
    "CIS_OTHER": 1800,
    "ISO": 1200,
    "CERTFR": 900,
    "ANSSI": 500,
    "NIST": 400,
    "OWASP": 100,
}
TRUSTED_FINANCE_SOURCE_RE = re.compile(
    r"(?:"
    r"^investments\.md$|"
    r"Monnaie, banque et marchés financiers|"
    r"Principes de l'économie|"
    r"Macroéconomie|"
    r"options-futures-and-other-derivatives|"
    r"valuation-measuring-and-managing|"
    r"financial-statement-analysis|"
    r"Security-Analysis|"
    r"Stochastic Calculus for Finance|"
    r"Financial Calculus|"
    r"trading-and-exchanges-market-microstructure|"
    r"Algorithmic and High-Frequency Trading|"
    r"advances-in-active-portfolio-management|"
    r"machine-learning-for-asset-managers|"
    r"Probability and Statistical Inference"
    r")",
    re.IGNORECASE,
)
CASE_SPECIFIC_RE = re.compile(
    r"\b(?:according to|in this simulation|in this example|in the example|"
    r"contribute(?:d|s)? to|reported|historically|the author|the paper|"
    r"this fund|this company|this trader)\b",
    re.IGNORECASE,
)


def load_cleaner():
    location = ROOT / "scripts" / "clean_training_jsonl.py"
    spec = importlib.util.spec_from_file_location("clean_training_jsonl", location)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CLEANER = load_cleaner()


@dataclass(frozen=True)
class Candidate:
    instruction: str
    response: str
    category: str
    score: int
    stable_key: str
    original_line: int
    source_file: str | None = None
    source_family: str | None = None
    source_sha256: str | None = None
    support_level: int | None = None
    support_coverage: float | None = None
    source_path: str | None = None
    chunk_index: int | None = None


def words(value: str, drop_stopwords: bool = False) -> list[str]:
    tokens = WORD_RE.findall(value.casefold())
    if drop_stopwords:
        return [token for token in tokens if token not in STOPWORDS]
    return tokens


def ngrams(tokens: list[str], size: int) -> set[str]:
    return {
        " ".join(tokens[index:index + size])
        for index in range(max(0, len(tokens) - size + 1))
    }


def stable_hash(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def near_duplicate_key(instruction: str) -> str:
    tokens = words(instruction, drop_stopwords=True)
    simplified = []
    for token in tokens:
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        simplified.append(token)
    return " ".join(sorted(simplified))


def category_for(text: str, categories: dict[str, re.Pattern[str]]) -> tuple[str, int]:
    matches = [(name, len(pattern.findall(text))) for name, pattern in categories.items()]
    name, count = max(matches, key=lambda item: item[1])
    return (name if count else "other"), count


def source_family(source_file: str) -> str | None:
    if re.search(r"(?:ANSSI|anssi|guide_hygiene|back to basics)", source_file):
        return "ANSSI"
    if source_file.startswith("NIST"):
        return "NIST"
    if source_file.startswith("CERTFR"):
        return "CERTFR"
    if "OWASP" in source_file:
        return "OWASP"
    if source_file.startswith("ISO_IEC"):
        return "ISO"
    if "CIS_AWS_Storage" in source_file:
        return "CIS_AWS_STORAGE"
    if source_file.startswith("CIS_"):
        return "CIS_OTHER"
    return None


def common_quality_reasons(instruction: str, response: str) -> list[str]:
    reasons = []
    normalized_instruction = CLEANER.normalize(instruction)
    normalized_response = CLEANER.normalize(response)
    if not 20 <= len(instruction) <= 220:
        reasons.append("instruction_length")
    if not 35 <= len(response) <= 900:
        reasons.append("response_length")
    if len(words(response)) < 7:
        reasons.append("response_too_short")
    if "\ufffd" in instruction or "\ufffd" in response:
        reasons.append("corrupted_unicode")
    if CLEANER.CONTEXT_DEPENDENT_RE.search(instruction):
        reasons.append("missing_source_context")
    if CLEANER.UNANCHORED_CURRENT_RE.search(instruction) and not CLEANER.DATE_ANCHOR_RE.search(instruction):
        reasons.append("unanchored_time_sensitive")
    if CLEANER.NONANSWER_RE.search(response):
        reasons.append("explicit_nonanswer")
    if META_RE.search(response):
        reasons.append("meta_response")
    if response.count("`") % 2:
        reasons.append("unbalanced_backticks")
    if normalized_instruction == normalized_response:
        reasons.append("answer_echoes_question")
    if response.endswith("?") and not CLEANER.QUESTION_REQUEST_RE.search(instruction):
        reasons.append("response_is_question")
    if normalized_instruction in CLEANER.KNOWN_BAD_INSTRUCTIONS:
        reasons.append("known_bad_answer")
    return reasons


def source_support(
    instruction: str,
    response: str,
    normalized_source: str,
) -> int:
    answer_tokens = words(response, drop_stopwords=True)
    for size, level in ((5, 3), (4, 2), (3, 1)):
        if any(group in normalized_source for group in ngrams(answer_tokens, size)):
            return level
    return 0


def source_support_coverage(response: str, normalized_source: str) -> float:
    answer_fourgrams = ngrams(words(response, drop_stopwords=True), 4)
    if not answer_fourgrams:
        return 0.0
    matches = sum(group in normalized_source for group in answer_fourgrams)
    return matches / len(answer_fourgrams)


def source_support_from_sets(
    response: str,
    source_fourgrams: set[tuple[str, ...]],
    source_fivegrams: set[tuple[str, ...]],
) -> tuple[int, float]:
    answer_tokens = words(response, drop_stopwords=True)
    answer_fourgrams = {
        tuple(answer_tokens[index:index + 4])
        for index in range(max(0, len(answer_tokens) - 3))
    }
    coverage = (
        sum(group in source_fourgrams for group in answer_fourgrams)
        / max(1, len(answer_fourgrams))
    )
    if any(
        tuple(answer_tokens[index:index + 5]) in source_fivegrams
        for index in range(max(0, len(answer_tokens) - 4))
    ):
        return 3, coverage
    if any(
        tuple(answer_tokens[index:index + 4]) in source_fourgrams
        for index in range(max(0, len(answer_tokens) - 3))
    ):
        return 2, coverage
    return 0, coverage


def build_pdf_index(pdf_root: Path) -> dict[str, Path]:
    index = {}
    for path in sorted(pdf_root.rglob("*")):
        if path.is_file() and path.suffix.casefold() == ".pdf":
            index.setdefault(path.stem.casefold(), path)
    return index


def load_pdf_source(
    source_file: str,
    pdf_index: dict[str, Path],
    source_cache: dict[str, tuple[str, str]],
) -> tuple[str, str, Path] | None:
    if source_file in source_cache:
        normalized, digest = source_cache[source_file]
        path = pdf_index[source_file.removesuffix(".md").casefold()]
        return normalized, digest, path
    stem = source_file.removesuffix(".md").casefold()
    path = pdf_index.get(stem)
    if path is None:
        return None
    with fitz.open(path) as document:
        content = "\n".join(page.get_text("text") for page in document)
    normalized = " ".join(words(content))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source_cache[source_file] = (normalized, digest)
    return normalized, digest, path


def cyber_candidates(
    raw_path: Path,
    pdf_root: Path,
) -> tuple[list[Candidate], collections.Counter[str]]:
    candidates = []
    rejected: collections.Counter[str] = collections.Counter()
    source_cache: dict[str, tuple[str, str]] = {}
    pdf_index = build_pdf_index(pdf_root)
    with raw_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            row = json.loads(raw)
            instruction = str(row.get("question", "")).strip()
            response = str(row.get("answer", "")).strip()
            source_file = str(row.get("_source_file", "")).strip()
            family = source_family(source_file)
            if family is None:
                rejected["non_official_source"] += 1
                continue
            if SOURCE_FAMILY_LIMITS[family] == 0:
                rejected["excluded_source_family"] += 1
                continue
            reasons = common_quality_reasons(instruction, response)
            combined = f"{instruction}\n{response}"
            if not CLEANER.CYBER_RE.search(combined):
                reasons.append("weak_domain_signal")
            loaded_source = load_pdf_source(source_file, pdf_index, source_cache)
            if loaded_source is None:
                reasons.append("source_file_missing")
                support = 0
                digest = None
            else:
                normalized_source, digest, source_path = loaded_source
                support = source_support(instruction, response, normalized_source)
                coverage = source_support_coverage(response, normalized_source)
                if not support:
                    reasons.append("not_textually_grounded")
                if coverage < 0.30:
                    reasons.append("low_textual_coverage")
            if reasons:
                for reason in set(reasons):
                    rejected[reason] += 1
                continue
            category, domain_hits = category_for(combined, CYBER_CATEGORIES)
            score = (
                support * 25
                + min(domain_hits, 5) * 4
                + (10 if CONCEPTUAL_QUESTION_RE.search(instruction) else 0)
                + (10 if 70 <= len(response) <= 450 else 0)
                + (5 if 35 <= len(instruction) <= 140 else 0)
                + round(coverage * 20)
            )
            candidates.append(
                Candidate(
                    instruction=instruction,
                    response=response,
                    category=category,
                    score=score,
                    stable_key=stable_hash(instruction, response, source_file),
                    original_line=line_number,
                    source_file=source_file,
                    source_family=family,
                    source_sha256=digest,
                    support_level=support,
                    support_coverage=coverage,
                    source_path=str(source_path),
                )
            )
    return candidates, rejected


def finance_candidates(
    raw_path: Path,
    pdf_root: Path,
) -> tuple[list[Candidate], collections.Counter[str]]:
    candidates = []
    rejected: collections.Counter[str] = collections.Counter()
    rows_by_source: dict[str, list[tuple[int, dict]]] = collections.defaultdict(list)
    pdf_index = build_pdf_index(pdf_root)
    with raw_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            row = json.loads(raw)
            rows_by_source[str(row.get("_source_file", "")).strip()].append(
                (line_number, row)
            )

    for source_file, source_rows in rows_by_source.items():
        if not TRUSTED_FINANCE_SOURCE_RE.search(source_file):
            rejected["source_outside_trusted_core"] += len(source_rows)
            continue
        source_path = pdf_index.get(source_file.removesuffix(".md").casefold())
        if source_path is None:
            rejected["source_file_missing"] += len(source_rows)
            continue
        source_bytes = source_path.read_bytes()
        with fitz.open(stream=source_bytes, filetype="pdf") as document:
            source_text = "\n".join(page.get_text("text") for page in document)
        source_tokens = words(source_text, drop_stopwords=True)
        source_fourgrams = {
            tuple(source_tokens[index:index + 4])
            for index in range(max(0, len(source_tokens) - 3))
        }
        source_fivegrams = {
            tuple(source_tokens[index:index + 5])
            for index in range(max(0, len(source_tokens) - 4))
        }
        digest = hashlib.sha256(source_bytes).hexdigest()

        for line_number, row in source_rows:
            instruction = str(row.get("question", "")).strip()
            response = str(row.get("answer", "")).strip()
            chunk_index = row.get("_chunk_idx")
            reasons = common_quality_reasons(instruction, response)
            combined = f"{instruction}\n{response}"
            if not CLEANER.FINANCE_RE.search(combined):
                reasons.append("weak_domain_signal")
            if TRIVIA_RE.search(instruction):
                reasons.append("biographical_or_historical_trivia")
            if not CONCEPTUAL_QUESTION_RE.search(instruction):
                reasons.append("not_conceptual")
            if FORMULA_HEAVY_RE.search(combined):
                reasons.append("formula_or_calculation")
            if PRECISE_CONTEXTUAL_FACT_RE.search(instruction):
                reasons.append("precise_contextual_fact")
            if re.search(r"\d", instruction):
                reasons.append("numeric_specific_question")
            if CASE_SPECIFIC_RE.search(instruction):
                reasons.append("case_specific_question")
            if re.search(
                r"\b(?:19|20)\d{2}\b|[$€£¥]|\b\d+(?:\.\d+)?\s*%",
                response,
            ):
                reasons.append("precise_numeric_answer")
            support, coverage = source_support_from_sets(
                response,
                source_fourgrams,
                source_fivegrams,
            )
            if support < 2:
                reasons.append("insufficient_textual_grounding")
            if coverage < 0.70:
                reasons.append("low_textual_coverage")
            if reasons:
                for reason in set(reasons):
                    rejected[reason] += 1
                continue
            category, domain_hits = category_for(combined, FINANCE_CATEGORIES)
            if category == "other":
                rejected["uncategorized"] += 1
                continue
            score = (
                min(domain_hits, 6) * 8
                + (20 if 80 <= len(response) <= 450 else 0)
                + (10 if 35 <= len(instruction) <= 150 else 0)
                + min(len(set(words(instruction, True)) & set(words(response, True))), 5) * 3
                + round(coverage * 20)
            )
            candidates.append(
                Candidate(
                    instruction=instruction,
                    response=response,
                    category=category,
                    score=score,
                    stable_key=stable_hash(instruction, response),
                    original_line=line_number,
                    source_file=source_file,
                    source_sha256=digest,
                    support_level=support,
                    support_coverage=coverage,
                    source_path=str(source_path),
                    chunk_index=chunk_index if isinstance(chunk_index, int) else None,
                )
            )
    return candidates, rejected


def select_diverse(
    candidates: list[Candidate],
    target: int,
    categories: dict[str, re.Pattern[str]],
    enforce_source_limits: bool,
) -> tuple[list[Candidate], collections.Counter[str]]:
    ordered = sorted(candidates, key=lambda item: (-item.score, item.stable_key))
    selected: list[Candidate] = []
    selected_keys: set[str] = set()
    question_keys: set[str] = set()
    response_counts: collections.Counter[str] = collections.Counter()
    source_counts: collections.Counter[str] = collections.Counter()
    family_counts: collections.Counter[str] = collections.Counter()
    category_counts: collections.Counter[str] = collections.Counter()
    skip_counts: collections.Counter[str] = collections.Counter()
    category_target = max(1, target // len(categories))

    def try_add(candidate: Candidate, category_limited: bool, family_limited: bool) -> bool:
        if candidate.stable_key in selected_keys:
            return False
        q_key = near_duplicate_key(candidate.instruction)
        if q_key in question_keys:
            skip_counts["near_duplicate_question"] += 1
            return False
        response_key = CLEANER.normalize(candidate.response)
        if response_counts[response_key] >= 2:
            skip_counts["repeated_response"] += 1
            return False
        if category_limited and category_counts[candidate.category] >= category_target:
            return False
        if candidate.source_file and source_counts[candidate.source_file] >= 300:
            skip_counts["source_cap"] += 1
            return False
        if family_limited and candidate.source_family:
            limit = SOURCE_FAMILY_LIMITS[candidate.source_family]
            if family_counts[candidate.source_family] >= limit:
                skip_counts["source_family_cap"] += 1
                return False
        selected.append(candidate)
        selected_keys.add(candidate.stable_key)
        question_keys.add(q_key)
        response_counts[response_key] += 1
        category_counts[candidate.category] += 1
        if candidate.source_file:
            source_counts[candidate.source_file] += 1
        if candidate.source_family:
            family_counts[candidate.source_family] += 1
        return True

    for candidate in ordered:
        if len(selected) >= target:
            break
        try_add(candidate, category_limited=True, family_limited=enforce_source_limits)
    for candidate in ordered:
        if len(selected) >= target:
            break
        try_add(candidate, category_limited=False, family_limited=enforce_source_limits)
    if not enforce_source_limits:
        for candidate in ordered:
            if len(selected) >= target:
                break
            try_add(candidate, category_limited=False, family_limited=False)

    selected.sort(key=lambda item: item.stable_key)
    return selected, skip_counts


def write_outputs(
    name: str,
    selected: list[Candidate],
    output_dir: Path,
    candidate_count: int,
    rejected: collections.Counter[str],
    dedupe_skips: collections.Counter[str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"{name}_gold.jsonl"
    provenance_path = output_dir / f"{name}_gold.provenance.jsonl"
    report_path = output_dir / f"{name}_gold.report.json"
    category_counts: collections.Counter[str] = collections.Counter()
    family_counts: collections.Counter[str] = collections.Counter()
    source_counts: collections.Counter[str] = collections.Counter()

    with (
        dataset_path.open("w", encoding="utf-8") as dataset,
        provenance_path.open("w", encoding="utf-8") as provenance,
    ):
        for gold_line, candidate in enumerate(selected, 1):
            text = (
                f"### Instruction:\n{candidate.instruction}"
                f"\n\n### Response:\n{candidate.response}"
            )
            dataset.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            record = {
                "gold_line": gold_line,
                "category": candidate.category,
                "quality_score": candidate.score,
                "original_line": candidate.original_line,
                "source_file": candidate.source_file,
                "source_family": candidate.source_family,
                "source_sha256": candidate.source_sha256,
                "source_path": candidate.source_path,
                "source_chunk_index": candidate.chunk_index,
                "textual_support_level": candidate.support_level,
                "textual_support_coverage": candidate.support_coverage,
                "source_status": "verified_local_document" if candidate.source_file else "unavailable",
            }
            provenance.write(json.dumps(record, ensure_ascii=False) + "\n")
            category_counts[candidate.category] += 1
            if candidate.source_family:
                family_counts[candidate.source_family] += 1
            if candidate.source_file:
                source_counts[candidate.source_file] += 1

    report = {
        "dataset": str(dataset_path),
        "provenance": str(provenance_path),
        "selected": len(selected),
        "eligible_candidates": candidate_count,
        "categories": dict(category_counts),
        "source_families": dict(family_counts),
        "top_sources": source_counts.most_common(25),
        "quality_rejections": dict(rejected),
        "diversity_rejections": dict(dedupe_skips),
        "provenance_warning": None,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cyber-target", type=int, default=4_000)
    parser.add_argument("--finance-target", type=int, default=5_000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "gold")
    parser.add_argument(
        "--cyber-raw",
        type=Path,
        default=ROOT / "data" / "results" / "dataset_qa_raw.jsonl",
    )
    parser.add_argument(
        "--cyber-pdf-dir",
        type=Path,
        default=ROOT / "data" / "source" / "00_DATA_SOURCE_LLM",
    )
    parser.add_argument(
        "--finance-raw",
        type=Path,
        default=ROOT.parent / "DATASETS" / "FINANCE" / "results" / "dataset_qa_raw.jsonl",
    )
    parser.add_argument(
        "--finance-pdf-dir",
        type=Path,
        default=ROOT.parent / "DATASETS" / "FINANCE" / "FINANCE_DOCS",
    )
    args = parser.parse_args()

    cyber_pool, cyber_rejected = cyber_candidates(args.cyber_raw, args.cyber_pdf_dir)
    finance_pool, finance_rejected = finance_candidates(
        args.finance_raw,
        args.finance_pdf_dir,
    )
    cyber_selected, cyber_dedupe = select_diverse(
        cyber_pool, args.cyber_target, CYBER_CATEGORIES, enforce_source_limits=True
    )
    finance_selected, finance_dedupe = select_diverse(
        finance_pool, args.finance_target, FINANCE_CATEGORIES, enforce_source_limits=False
    )
    reports = [
        write_outputs(
            "cyber", cyber_selected, args.output_dir, len(cyber_pool),
            cyber_rejected, cyber_dedupe,
        ),
        write_outputs(
            "finance", finance_selected, args.output_dir, len(finance_pool),
            finance_rejected, finance_dedupe,
        ),
    ]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
