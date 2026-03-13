"""Contextual compression for retrieved chunks.

After retrieval, many chunks contain only *partially* relevant content.
The compressors in this module filter or compress each chunk to retain
only the sentences (or spans) that are actually relevant to the query.

This reduces token count, lowers LLM hallucination risk, and can improve
faithfulness metrics — all without any external model dependency.

Two compressors are provided:

SentenceCompressor
    Splits each chunk into sentences, scores each sentence against the query
    using TF-IDF-style token overlap, and returns only sentences that exceed
    a relevance threshold.  Falls back to the full chunk if nothing passes.

KeywordCompressor
    Uses a simpler keyword-presence filter.  Fast and deterministic;
    useful as a lightweight alternative or for testing.

Both implement the :class:`BaseCompressor` interface:

    ``compress(query, chunks) -> list[CompressedChunk]``

Typical usage::

    from lexisearch.retrieval.compression import SentenceCompressor
    from lexisearch.retrieval.advanced import RetrievedChunk

    compressor = SentenceCompressor(threshold=0.15, min_sentences=1)
    compressed = compressor.compress(query, retrieved_chunks)
    for c in compressed:
        print(c.compressed_content)
"""

from __future__ import annotations

import logging
import math
import re
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Sentence boundary pattern — handles common abbreviations imperfectly but
# correctly in the vast majority of legal text.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class CompressedChunk:
    """A chunk after contextual compression."""

    chunk_id: str
    original_content: str
    compressed_content: str
    compression_ratio: float
    """len(compressed) / len(original), in [0.0, 1.0]."""
    relevance_score: float
    """Aggregate relevance score of the retained sentences."""
    metadata: dict[str, Any] = field(default_factory=dict)
    retained_sentences: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseCompressor:
    """Abstract base for contextual compressors."""

    @abstractmethod
    def compress(
        self,
        query: str,
        chunks: list[Any],
    ) -> list[CompressedChunk]:
        """Compress *chunks* in the context of *query*.

        Parameters
        ----------
        query:
            The user query — used to score sentence relevance.
        chunks:
            List of :class:`~lexisearch.retrieval.advanced.RetrievedChunk`
            (or any object with ``.chunk_id``, ``.content``, ``.metadata``).

        Returns:
        -------
        list[CompressedChunk]
            One compressed result per input chunk (same order preserved).
        """


# ---------------------------------------------------------------------------
# TF-IDF token scorer
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, length >= 2."""
    return [t for t in re.findall(r"\b[a-z]{2,}\b", text.lower())]


_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "of",
        "in",
        "to",
        "for",
        "on",
        "at",
        "with",
        "by",
        "from",
        "and",
        "or",
        "but",
        "if",
        "that",
        "this",
        "it",
        "its",
        "not",
        "no",
        "as",
        "so",
    }
)


def _query_keywords(query: str) -> set[str]:
    """Extract meaningful keywords from the query."""
    return {t for t in _tokenise(query) if t not in _STOP_WORDS}


def _sentence_score(sentence: str, query_kws: set[str]) -> float:
    """Return a relevance score in [0.0, 1.0] for *sentence* against *query_kws*.

    Uses Jaccard-like overlap between sentence keywords and query keywords,
    adjusted by a log-normalised length penalty to avoid very short sentences
    always winning.
    """
    if not query_kws:
        return 0.0
    sent_tokens = set(_tokenise(sentence)) - _STOP_WORDS
    if not sent_tokens:
        return 0.0
    overlap = len(query_kws & sent_tokens)
    union = len(query_kws | sent_tokens)
    jaccard = overlap / union if union else 0.0
    # Mild length bonus: longer sentences cover more ground
    length_bonus = math.log(max(len(sent_tokens), 1) + 1) / 5.0
    return min(jaccard + length_bonus * 0.1, 1.0)


# ---------------------------------------------------------------------------
# SentenceCompressor
# ---------------------------------------------------------------------------


class SentenceCompressor(BaseCompressor):
    """Compress chunks by retaining only relevant sentences.

    Parameters
    ----------
    threshold:
        Minimum relevance score for a sentence to be retained.
        Scores are in [0.0, 1.0].  Default 0.12 is deliberately low to
        avoid discarding borderline-relevant sentences.
    min_sentences:
        Always retain at least this many sentences (by score) even if none
        exceed *threshold*.  Prevents returning empty compressed chunks.
    max_sentences:
        Hard upper limit on retained sentences per chunk.  ``None`` means
        no limit.
    join_sep:
        Separator used when joining retained sentences.
    """

    def __init__(
        self,
        threshold: float = 0.12,
        min_sentences: int = 1,
        max_sentences: int | None = None,
        join_sep: str = " … ",
    ) -> None:
        """Initialise SentenceCompressor with filtering parameters."""
        self.threshold = threshold
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.join_sep = join_sep

    def compress(self, query: str, chunks: list[Any]) -> list[CompressedChunk]:
        """Compress *chunks* by retaining only query-relevant sentences."""
        query_kws = _query_keywords(query)
        results: list[CompressedChunk] = []

        for chunk in chunks:
            chunk_id = getattr(chunk, "chunk_id", "") or ""
            content = getattr(chunk, "content", "") or ""
            metadata = dict(getattr(chunk, "metadata", {}) or {})

            compressed = self._compress_text(content, query_kws)
            ratio = len(compressed) / max(len(content), 1)
            retained = self._split_sentences(compressed)

            results.append(
                CompressedChunk(
                    chunk_id=chunk_id,
                    original_content=content,
                    compressed_content=compressed,
                    compression_ratio=ratio,
                    relevance_score=self._mean_score(retained, query_kws),
                    metadata=metadata,
                    retained_sentences=retained,
                )
            )
            logger.debug(
                "Compressed chunk %r: %.0f%% retained (%d → %d chars)",
                chunk_id,
                ratio * 100,
                len(content),
                len(compressed),
            )

        return results

    def _compress_text(self, content: str, query_kws: set[str]) -> str:
        """Return the compressed text for one chunk."""
        sentences = self._split_sentences(content)
        if not sentences:
            return content

        scored = [(s, _sentence_score(s, query_kws)) for s in sentences]
        above_threshold = [(s, sc) for s, sc in scored if sc >= self.threshold]

        if len(above_threshold) < self.min_sentences:
            # Fall back to top-N by score
            above_threshold = sorted(scored, key=lambda x: x[1], reverse=True)[: self.min_sentences]
            # Restore original order
            order = {s: i for i, (s, _) in enumerate(scored)}
            above_threshold.sort(key=lambda x: order.get(x[0], 0))

        if self.max_sentences is not None:
            above_threshold = above_threshold[: self.max_sentences]

        return self.join_sep.join(s for s, _ in above_threshold)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split *text* into sentences."""
        parts = _SENTENCE_RE.split(text.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _mean_score(sentences: list[str], query_kws: set[str]) -> float:
        if not sentences:
            return 0.0
        scores = [_sentence_score(s, query_kws) for s in sentences]
        return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# KeywordCompressor
# ---------------------------------------------------------------------------


class KeywordCompressor(BaseCompressor):
    """Compress chunks by retaining sentences that contain query keywords.

    Simpler and faster than :class:`SentenceCompressor` — useful as a
    lightweight baseline or when exact keyword matching is preferred over
    Jaccard scoring.

    Parameters
    ----------
    min_keyword_matches:
        A sentence is kept if at least this many query keywords appear in it.
    min_sentences:
        Always keep at least this many sentences (best-matching) per chunk.
    """

    def __init__(
        self,
        min_keyword_matches: int = 1,
        min_sentences: int = 1,
        join_sep: str = " … ",
    ) -> None:
        """Initialise KeywordCompressor."""
        self.min_keyword_matches = min_keyword_matches
        self.min_sentences = min_sentences
        self.join_sep = join_sep

    def compress(self, query: str, chunks: list[Any]) -> list[CompressedChunk]:
        """Compress *chunks* by keyword presence filtering."""
        query_kws = _query_keywords(query)
        results: list[CompressedChunk] = []

        for chunk in chunks:
            chunk_id = getattr(chunk, "chunk_id", "") or ""
            content = getattr(chunk, "content", "") or ""
            metadata = dict(getattr(chunk, "metadata", {}) or {})

            sentences = [s.strip() for s in _SENTENCE_RE.split(content.strip()) if s.strip()]
            retained = self._filter(sentences, query_kws)

            compressed = self.join_sep.join(retained) if retained else content
            ratio = len(compressed) / max(len(content), 1)

            results.append(
                CompressedChunk(
                    chunk_id=chunk_id,
                    original_content=content,
                    compressed_content=compressed,
                    compression_ratio=ratio,
                    relevance_score=float(len(retained)) / max(len(sentences), 1),
                    metadata=metadata,
                    retained_sentences=retained,
                )
            )

        return results

    def _filter(self, sentences: list[str], query_kws: set[str]) -> list[str]:
        """Return sentences with >= min_keyword_matches query keywords."""
        if not query_kws:
            return sentences[: self.min_sentences]

        def _count(s: str) -> int:
            tokens = set(_tokenise(s)) - _STOP_WORDS
            return len(query_kws & tokens)

        passing = [s for s in sentences if _count(s) >= self.min_keyword_matches]

        if len(passing) < self.min_sentences:
            ranked = sorted(sentences, key=_count, reverse=True)
            passing = ranked[: self.min_sentences]
            # Restore original order
            order = {s: i for i, s in enumerate(sentences)}
            passing.sort(key=lambda s: order.get(s, 0))

        return passing
