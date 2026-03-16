"""Document deduplication for search results.

Provides multiple strategies for detecting and removing near-duplicate
results from retrieval output — critical for legal search where the same
case may appear with minor text variations (formatting, whitespace,
encoding differences).

Deduplication strategies::

    BaseDeduplicator
    ├── ExactDeduplicator      (normalised exact match)
    ├── SimHashDeduplicator    (locality-sensitive fingerprinting)
    └── MinHashDeduplicator    (set-similarity via Jaccard estimation)

Usage::

    from lexisearch.retrieval.dedup import (
        DeduplicationPipeline, SimHashDeduplicator, ExactDeduplicator
    )

    pipeline = DeduplicationPipeline([
        ExactDeduplicator(),
        SimHashDeduplicator(hamming_threshold=3),
    ])
    result = pipeline.deduplicate(search_results)
    print(f"Removed {result.stats.duplicates} duplicates")
"""

from __future__ import annotations

import hashlib
import re
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lexisearch.models import SearchResult

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DedupStats:
    """Statistics from a deduplication run.

    Attributes:
        total: Number of results before dedup.
        unique: Number of results after dedup.
        duplicates: Number of results removed.
        dedup_time_ms: Wall-clock time in milliseconds.
    """

    total: int
    unique: int
    duplicates: int
    dedup_time_ms: float


@dataclass
class DuplicateGroup:
    """A group of results identified as duplicates of each other.

    Attributes:
        canonical_index: Index of the representative result kept.
        duplicate_indices: Indices of the results removed.
        similarity: Estimated similarity score (1.0 = exact match).
    """

    canonical_index: int
    duplicate_indices: list[int] = field(default_factory=list)
    similarity: float = 1.0


@dataclass
class DedupResult:
    """Result of a deduplication operation.

    Attributes:
        original: The input results (unchanged).
        deduplicated: Results with duplicates removed, preserving order.
        duplicate_groups: Groups of duplicates found.
        stats: Summary statistics.
    """

    original: list[SearchResult]
    deduplicated: list[SearchResult]
    duplicate_groups: list[DuplicateGroup]
    stats: DedupStats


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Normalise text for comparison: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _shingles(text: str, k: int = 3) -> set[str]:
    """Generate character k-shingles (k-grams) from text."""
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _token_shingles(text: str, k: int = 2) -> set[str]:
    """Generate word-level k-shingles from text."""
    words = text.split()
    if len(words) < k:
        return {text} if text else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseDeduplicator(ABC):
    """Abstract base for deduplication strategies."""

    @abstractmethod
    def deduplicate(self, results: Sequence[SearchResult]) -> DedupResult:
        """Remove near-duplicate results.

        Args:
            results: Search results to deduplicate.

        Returns:
            A DedupResult with deduplicated results and metadata.
        """


# ---------------------------------------------------------------------------
# Exact deduplication
# ---------------------------------------------------------------------------


class ExactDeduplicator(BaseDeduplicator):
    """Remove results with identical normalised text.

    This is the fastest strategy and catches exact duplicates that
    differ only in whitespace, punctuation, or casing.
    """

    def deduplicate(self, results: Sequence[SearchResult]) -> DedupResult:
        """Remove results whose normalised text matches exactly."""
        start = time.perf_counter()
        results_list = list(results)

        seen: dict[str, int] = {}
        kept: list[SearchResult] = []
        groups: list[DuplicateGroup] = []
        kept_indices: list[int] = []

        for i, result in enumerate(results_list):
            content = _normalise(result.chunk.content)
            fingerprint = hashlib.md5(content.encode("utf-8")).hexdigest()

            if fingerprint in seen:
                canonical = seen[fingerprint]
                # Find or create group
                group = None
                for g in groups:
                    if g.canonical_index == canonical:
                        group = g
                        break
                if group is None:
                    group = DuplicateGroup(canonical_index=canonical, similarity=1.0)
                    groups.append(group)
                group.duplicate_indices.append(i)
            else:
                seen[fingerprint] = i
                kept.append(result)
                kept_indices.append(i)

        elapsed = (time.perf_counter() - start) * 1000
        return DedupResult(
            original=results_list,
            deduplicated=kept,
            duplicate_groups=groups,
            stats=DedupStats(
                total=len(results_list),
                unique=len(kept),
                duplicates=len(results_list) - len(kept),
                dedup_time_ms=elapsed,
            ),
        )


# ---------------------------------------------------------------------------
# SimHash deduplication
# ---------------------------------------------------------------------------


def _simhash(text: str, hashbits: int = 64) -> int:
    """Compute a SimHash fingerprint for the given text.

    SimHash is a locality-sensitive hash: similar documents produce
    fingerprints with small Hamming distance.
    """
    tokens = _shingles(_normalise(text), k=3)
    vector = [0] * hashbits

    for token in tokens:
        token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(hashbits):
            if token_hash & (1 << i):
                vector[i] += 1
            else:
                vector[i] -= 1

    fingerprint = 0
    for i in range(hashbits):
        if vector[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two integers."""
    xor = a ^ b
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count


class SimHashDeduplicator(BaseDeduplicator):
    """Detect near-duplicates via SimHash fingerprint comparison.

    SimHash produces a fixed-size fingerprint where similar documents
    have small Hamming distance. This enables O(n²) pairwise comparison
    but with very fast per-pair checks.

    Args:
        hamming_threshold: Maximum Hamming distance to consider two
            documents as duplicates. Lower = stricter. Default 3.
        hashbits: Number of bits in the SimHash fingerprint. Default 64.
    """

    def __init__(self, hamming_threshold: int = 3, hashbits: int = 64) -> None:
        """Initialise with hamming distance threshold and hash size."""
        self.hamming_threshold = hamming_threshold
        self.hashbits = hashbits

    def deduplicate(self, results: Sequence[SearchResult]) -> DedupResult:
        """Remove results whose SimHash fingerprints are within hamming threshold."""
        start = time.perf_counter()
        results_list = list(results)

        if not results_list:
            elapsed = (time.perf_counter() - start) * 1000
            return DedupResult(
                original=[],
                deduplicated=[],
                duplicate_groups=[],
                stats=DedupStats(0, 0, 0, elapsed),
            )

        # Compute fingerprints
        fingerprints = [_simhash(r.chunk.content, self.hashbits) for r in results_list]

        # Track which indices are duplicates
        is_duplicate = [False] * len(results_list)
        groups: list[DuplicateGroup] = []

        for i in range(len(results_list)):
            if is_duplicate[i]:
                continue
            group_dupes: list[int] = []
            for j in range(i + 1, len(results_list)):
                if is_duplicate[j]:
                    continue
                dist = _hamming_distance(fingerprints[i], fingerprints[j])
                if dist <= self.hamming_threshold:
                    is_duplicate[j] = True
                    group_dupes.append(j)
            if group_dupes:
                groups.append(
                    DuplicateGroup(
                        canonical_index=i,
                        duplicate_indices=group_dupes,
                        similarity=1.0 - (self.hamming_threshold / self.hashbits),
                    )
                )

        kept = [r for i, r in enumerate(results_list) if not is_duplicate[i]]
        elapsed = (time.perf_counter() - start) * 1000

        return DedupResult(
            original=results_list,
            deduplicated=kept,
            duplicate_groups=groups,
            stats=DedupStats(
                total=len(results_list),
                unique=len(kept),
                duplicates=sum(is_duplicate),
                dedup_time_ms=elapsed,
            ),
        )


# ---------------------------------------------------------------------------
# MinHash deduplication
# ---------------------------------------------------------------------------


def _minhash_signature(shingle_set: set[str], num_perm: int = 128) -> list[int]:
    """Compute a MinHash signature for a set of shingles.

    Uses random hash functions (seeded by permutation index) to estimate
    Jaccard similarity between sets.
    """
    max_hash = (1 << 32) - 1
    signature = [max_hash] * num_perm

    for shingle in shingle_set:
        shingle_bytes = shingle.encode("utf-8")
        for i in range(num_perm):
            # Use MD5 with seed for deterministic "random" hash
            seed = struct.pack("<I", i)
            h = int(hashlib.md5(seed + shingle_bytes).hexdigest()[:8], 16)
            if h < signature[i]:
                signature[i] = h

    return signature


def _jaccard_from_minhash(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b, strict=False) if a == b)
    return matches / len(sig_a)


class MinHashDeduplicator(BaseDeduplicator):
    """Detect near-duplicates via MinHash Jaccard similarity estimation.

    MinHash estimates the Jaccard similarity of two document shingle
    sets using compact signatures, enabling efficient fuzzy matching.

    Args:
        threshold: Minimum estimated Jaccard similarity to consider
            two documents as duplicates. Default 0.8 (80% similar).
        num_perm: Number of hash permutations (higher = more accurate
            but slower). Default 128.
        shingle_k: Size of word-level shingles. Default 2.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        shingle_k: int = 2,
    ) -> None:
        """Initialise with Jaccard threshold, permutation count, and shingle size."""
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_k = shingle_k

    def deduplicate(self, results: Sequence[SearchResult]) -> DedupResult:
        """Remove results whose MinHash signatures exceed the Jaccard threshold."""
        start = time.perf_counter()
        results_list = list(results)

        if not results_list:
            elapsed = (time.perf_counter() - start) * 1000
            return DedupResult(
                original=[],
                deduplicated=[],
                duplicate_groups=[],
                stats=DedupStats(0, 0, 0, elapsed),
            )

        # Compute shingles and signatures
        signatures = []
        for r in results_list:
            normalised = _normalise(r.chunk.content)
            shingles = _token_shingles(normalised, self.shingle_k)
            sig = _minhash_signature(shingles, self.num_perm)
            signatures.append(sig)

        # Pairwise comparison
        is_duplicate = [False] * len(results_list)
        groups: list[DuplicateGroup] = []

        for i in range(len(results_list)):
            if is_duplicate[i]:
                continue
            group_dupes: list[int] = []
            for j in range(i + 1, len(results_list)):
                if is_duplicate[j]:
                    continue
                jaccard = _jaccard_from_minhash(signatures[i], signatures[j])
                if jaccard >= self.threshold:
                    is_duplicate[j] = True
                    group_dupes.append(j)
            if group_dupes:
                groups.append(
                    DuplicateGroup(
                        canonical_index=i,
                        duplicate_indices=group_dupes,
                        similarity=self.threshold,
                    )
                )

        kept = [r for i, r in enumerate(results_list) if not is_duplicate[i]]
        elapsed = (time.perf_counter() - start) * 1000

        return DedupResult(
            original=results_list,
            deduplicated=kept,
            duplicate_groups=groups,
            stats=DedupStats(
                total=len(results_list),
                unique=len(kept),
                duplicates=sum(is_duplicate),
                dedup_time_ms=elapsed,
            ),
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DeduplicationPipeline(BaseDeduplicator):
    """Chain multiple deduplication strategies in sequence.

    Each strategy operates on the output of the previous one, allowing
    progressively finer deduplication (e.g., exact match first, then
    fuzzy matching on the remaining results).

    Args:
        strategies: Ordered list of deduplicators to apply.

    Example::

        pipeline = DeduplicationPipeline([
            ExactDeduplicator(),
            SimHashDeduplicator(hamming_threshold=3),
        ])
        result = pipeline.deduplicate(search_results)
    """

    def __init__(self, strategies: Sequence[BaseDeduplicator] | None = None) -> None:
        """Initialise with an ordered list of deduplication strategies."""
        self.strategies = list(strategies) if strategies else []

    def add(self, strategy: BaseDeduplicator) -> DeduplicationPipeline:
        """Add a strategy to the pipeline. Returns self for chaining."""
        self.strategies.append(strategy)
        return self

    def deduplicate(self, results: Sequence[SearchResult]) -> DedupResult:
        """Apply all strategies in sequence, deduplicating progressively."""
        start = time.perf_counter()
        results_list = list(results)
        current = results_list
        all_groups: list[DuplicateGroup] = []
        total_removed = 0

        for strategy in self.strategies:
            step_result = strategy.deduplicate(current)
            current = step_result.deduplicated
            all_groups.extend(step_result.duplicate_groups)
            total_removed += step_result.stats.duplicates

        elapsed = (time.perf_counter() - start) * 1000
        return DedupResult(
            original=results_list,
            deduplicated=current,
            duplicate_groups=all_groups,
            stats=DedupStats(
                total=len(results_list),
                unique=len(current),
                duplicates=len(results_list) - len(current),
                dedup_time_ms=elapsed,
            ),
        )
