"""Near-duplicate removal for generated Q&A pairs."""

from __future__ import annotations

import hashlib
import difflib
from collections import defaultdict

from tqdm import tqdm


def _get_ngrams(text, n=3):
    """Generate character n-grams (shingles) from text."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _minhash_signature(shingles, num_hashes=64):
    """Compute a reproducible MinHash signature for a set of shingles.

    Python's built-in string hash is randomized between processes, which made
    the LSH candidate set vary between runs. BLAKE2 creates stable base hashes;
    SplitMix64-style mixing cheaply derives deterministic permutations.
    """
    if not shingles:
        return tuple(0 for _ in range(num_hashes))
    mask = 0xFFFFFFFFFFFFFFFF
    base_hashes = [
        int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for shingle in shingles
    ]
    signature = []
    for i in range(num_hashes):
        min_hash = float('inf')
        seed = (0x9E3779B97F4A7C15 * (i + 1)) & mask
        for base_hash in base_hashes:
            h = (base_hash + seed) & mask
            h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & mask
            h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & mask
            h ^= h >> 31
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return tuple(signature)


def _lsh_buckets(signature, num_bands=16):
    """Split a MinHash signature into bands for LSH bucketing."""
    band_size = len(signature) // num_bands
    bands = []
    for i in range(num_bands):
        band = signature[i * band_size:(i + 1) * band_size]
        payload = b"".join(value.to_bytes(8, "big") for value in band)
        bands.append(
            int.from_bytes(
                hashlib.blake2b(payload, digest_size=8).digest(),
                "big",
            )
        )
    return bands


def deduplicate_qa(qa_list, threshold=0.85):
    if not qa_list:
        return []

    num_hashes = 64
    num_bands = 16

    # Phase 1: Exact dedup via normalized text hash
    seen_exact = {}
    deduped_exact = []
    for qa in qa_list:
        key = qa.get("question", "").lower().strip()
        if key not in seen_exact:
            seen_exact[key] = True
            deduped_exact.append(qa)

    exact_removed = len(qa_list) - len(deduped_exact)
    if exact_removed:
        tqdm.write(f"  Removed {exact_removed} exact duplicates.")

    # Phase 2: Build MinHash signatures + LSH index (cached)
    print(f"  Building similarity index for {len(deduped_exact)} pairs...")
    signatures = []
    all_bands = []  # pre-cache LSH bands to avoid recomputing in phase 3
    for qa in tqdm(deduped_exact, desc="Indexing", unit="pair"):
        shingles = _get_ngrams(qa.get("question", ""), n=3)
        sig = _minhash_signature(shingles, num_hashes)
        signatures.append(sig)
        all_bands.append(_lsh_buckets(sig, num_bands))

    # Build LSH band buckets: (band_idx, band_hash) -> list of indices
    band_buckets = defaultdict(list)
    for idx, bands in enumerate(all_bands):
        for band_idx, band_hash in enumerate(bands):
            band_buckets[(band_idx, band_hash)].append(idx)

    # Phase 3: For each pair, check only LSH candidates with SequenceMatcher
    removed = set()
    for idx in tqdm(range(len(deduped_exact)), desc="Deduplicating", unit="pair"):
        if idx in removed:
            continue

        # Collect candidate indices from shared LSH bands (using cached bands)
        candidates = set()
        for band_idx, band_hash in enumerate(all_bands[idx]):
            for cand_idx in band_buckets[(band_idx, band_hash)]:
                if cand_idx > idx and cand_idx not in removed:
                    candidates.add(cand_idx)

        if not candidates:
            continue

        q_text = deduped_exact[idx].get("question", "").lower()
        for cand_idx in candidates:
            if cand_idx in removed:
                continue
            cand_text = deduped_exact[cand_idx].get("question", "").lower()
            if difflib.SequenceMatcher(None, q_text, cand_text).ratio() > threshold:
                removed.add(cand_idx)

    unique_qa = [qa for idx, qa in enumerate(deduped_exact) if idx not in removed]
    return unique_qa
