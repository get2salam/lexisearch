"""Retrieval explainability — score attribution and chunk provenance.

After retrieval, it is often unclear *why* a chunk received its final score.
This module provides tools to decompose and explain retrieval decisions:

ChunkExplanation
    Per-chunk breakdown showing which query variants contributed, what
    terms overlap with the query, and a human-readable rationale string.

RetrievalExplainer
    Wraps a :class:`~lexisearch.retrieval.advanced.AdvancedRetrievalResult`
    and produces a :class:`RetrievalExplanation` with per-chunk details.

Typical usage::

    from lexisearch.retrieval.explain import RetrievalExplainer
    from lexisearch.retrieval import MultiQueryRetriever, AdvancedRetrievalConfig

    retriever = MultiQueryRetriever(my_base_retriever)
    result = retriever.retrieve("What is consideration in contract law?", top_k=5)

    explainer = RetrievalExplainer()
    explanation = explainer.explain(result)

    for chunk_exp in explanation.chunks:
        print(chunk_exp.summary())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Token utilities (shared with compression module conceptually)
# ---------------------------------------------------------------------------

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


def _keywords(text: str) -> set[str]:
    """Extract lowercase non-stopword tokens (len >= 2)."""
    return {t for t in re.findall(r"\b[a-z]{2,}\b", text.lower()) if t not in _STOP_WORDS}


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class TermOverlap:
    """Token-level overlap between a chunk and the query."""

    query_terms: list[str]
    """All meaningful terms in the query."""
    chunk_terms: list[str]
    """All meaningful terms in the chunk content."""
    matched_terms: list[str]
    """Terms that appear in both query and chunk."""
    overlap_ratio: float
    """len(matched) / len(query_terms), in [0.0, 1.0]."""


@dataclass
class SubQueryContribution:
    """How much a specific sub-query contributed to this chunk's final score."""

    sub_query: str
    rank: int
    """Rank of this chunk in the sub-query's result list (1-based, 0 = absent)."""
    rrf_contribution: float
    """Contribution to the RRF aggregate score from this sub-query."""


@dataclass
class ChunkExplanation:
    """Explanation for a single retrieved chunk."""

    chunk_id: str
    content_preview: str
    """First 200 characters of the chunk content."""
    final_score: float
    """The score as returned by the retriever."""
    strategy: str
    """The retrieval strategy that produced this chunk (e.g. 'multi_query')."""
    term_overlap: TermOverlap
    sub_query_contributions: list[SubQueryContribution] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable one-line summary of the explanation."""
        matched = ", ".join(self.term_overlap.matched_terms[:5])
        if len(self.term_overlap.matched_terms) > 5:
            matched += f" (+{len(self.term_overlap.matched_terms) - 5} more)"
        return (
            f"[{self.chunk_id}] score={self.final_score:.4f} "
            f"overlap={self.term_overlap.overlap_ratio:.0%} "
            f"({len(self.term_overlap.matched_terms)} matched terms: {matched or 'none'})"
        )

    def rationale(self) -> str:
        """Return a multi-line rationale string suitable for debugging."""
        lines = [
            f"Chunk: {self.chunk_id}",
            f"Strategy: {self.strategy}",
            f"Final score: {self.final_score:.6f}",
            "",
            "Term overlap:",
            f"  Query terms ({len(self.term_overlap.query_terms)}): "
            f"{', '.join(self.term_overlap.query_terms[:10])}",
            f"  Matched ({len(self.term_overlap.matched_terms)}): "
            f"{', '.join(self.term_overlap.matched_terms) or '(none)'}",
            f"  Overlap ratio: {self.term_overlap.overlap_ratio:.1%}",
        ]
        if self.sub_query_contributions:
            lines += ["", "Sub-query contributions:"]
            for contrib in sorted(
                self.sub_query_contributions, key=lambda c: c.rrf_contribution, reverse=True
            ):
                rank_str = f"rank {contrib.rank}" if contrib.rank > 0 else "not retrieved"
                lines.append(
                    f"  [{rank_str}] +{contrib.rrf_contribution:.4f} "
                    f"from: {contrib.sub_query[:60]!r}"
                )
        lines += [
            "",
            f"Content preview: {self.content_preview[:120]}...",
        ]
        return "\n".join(lines)


@dataclass
class RetrievalExplanation:
    """Explanation for a complete retrieval result."""

    query: str
    strategy: str
    sub_queries: list[str]
    chunks: list[ChunkExplanation]
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a tabular summary of all chunk explanations."""
        header = (
            f"Query: {self.query!r}\n"
            f"Strategy: {self.strategy} | Sub-queries: {len(self.sub_queries)}\n"
            f"{'─' * 72}\n"
        )
        rows = "\n".join(f"  {i + 1}. {c.summary()}" for i, c in enumerate(self.chunks))
        return header + rows


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------


class RetrievalExplainer:
    """Produce human-readable explanations for retrieval results.

    Parameters
    ----------
    rrf_k:
        The RRF constant used during retrieval.  Needed to reconstruct
        per-sub-query RRF contributions.  Should match the value used
        in the retriever's :class:`~lexisearch.retrieval.advanced.AdvancedRetrievalConfig`.
    preview_length:
        Number of characters to include in ``content_preview``.
    """

    def __init__(self, rrf_k: int = 60, preview_length: int = 200) -> None:
        """Initialise the explainer."""
        self.rrf_k = rrf_k
        self.preview_length = preview_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        result: Any,
        *,
        sub_query_results: list[list[Any]] | None = None,
    ) -> RetrievalExplanation:
        """Produce an explanation for a retrieval result.

        Parameters
        ----------
        result:
            An :class:`~lexisearch.retrieval.advanced.AdvancedRetrievalResult`
            (or any object with ``.query``, ``.chunks``, ``.strategy``,
            ``.sub_queries`` attributes).
        sub_query_results:
            Optional list of per-sub-query ranked chunk lists.  When
            supplied, per-sub-query RRF contributions are computed for each
            chunk.  When ``None``, contribution attribution is skipped.

        Returns:
        -------
        RetrievalExplanation
        """
        query = getattr(result, "query", "")
        strategy = getattr(result, "strategy", "unknown")
        sub_queries = list(getattr(result, "sub_queries", []))
        chunks = list(getattr(result, "chunks", []))
        metadata = dict(getattr(result, "metadata", {}))

        query_kws = sorted(_keywords(query))
        chunk_explanations: list[ChunkExplanation] = []

        for chunk in chunks:
            chunk_id = getattr(chunk, "chunk_id", "")
            content = getattr(chunk, "content", "")
            score = float(getattr(chunk, "score", 0.0))
            chunk_meta = dict(getattr(chunk, "metadata", {}))

            term_overlap = self._compute_term_overlap(query_kws, content)
            contributions = self._compute_contributions(
                chunk_id, content, sub_query_results or [], sub_queries
            )

            chunk_explanations.append(
                ChunkExplanation(
                    chunk_id=chunk_id,
                    content_preview=content[: self.preview_length],
                    final_score=score,
                    strategy=strategy,
                    term_overlap=term_overlap,
                    sub_query_contributions=contributions,
                    metadata=chunk_meta,
                )
            )

        return RetrievalExplanation(
            query=query,
            strategy=strategy,
            sub_queries=sub_queries,
            chunks=chunk_explanations,
            metadata=metadata,
        )

    def explain_score(self, query: str, chunk_content: str, score: float) -> ChunkExplanation:
        """Explain a single (query, chunk, score) triple without a full result.

        Useful for spot-checking why a specific chunk received a given score.
        """
        query_kws = sorted(_keywords(query))
        term_overlap = self._compute_term_overlap(query_kws, chunk_content)
        return ChunkExplanation(
            chunk_id="",
            content_preview=chunk_content[: self.preview_length],
            final_score=score,
            strategy="unknown",
            term_overlap=term_overlap,
            sub_query_contributions=[],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_term_overlap(self, query_kws: list[str], content: str) -> TermOverlap:
        """Compute token-level overlap between the query and chunk content."""
        chunk_kws = _keywords(content)
        query_set = set(query_kws)
        matched = sorted(query_set & chunk_kws)
        ratio = len(matched) / len(query_set) if query_set else 0.0
        return TermOverlap(
            query_terms=query_kws,
            chunk_terms=sorted(chunk_kws),
            matched_terms=matched,
            overlap_ratio=ratio,
        )

    def _compute_contributions(
        self,
        chunk_id: str,
        content: str,
        sub_query_results: list[list[Any]],
        sub_queries: list[str],
    ) -> list[SubQueryContribution]:
        """Compute per-sub-query RRF contributions for this chunk."""
        if not sub_query_results:
            return []

        contributions: list[SubQueryContribution] = []
        for sq_idx, sq_result in enumerate(sub_query_results):
            sq_label = sub_queries[sq_idx] if sq_idx < len(sub_queries) else f"sub_query_{sq_idx}"
            rank = self._find_rank(chunk_id, content, sq_result)
            rrf_contribution = 1.0 / (self.rrf_k + rank) if rank > 0 else 0.0
            contributions.append(
                SubQueryContribution(
                    sub_query=sq_label,
                    rank=rank,
                    rrf_contribution=rrf_contribution,
                )
            )
        return contributions

    @staticmethod
    def _find_rank(chunk_id: str, content: str, result_list: list[Any]) -> int:
        """Return the 1-based rank of chunk in result_list, or 0 if not found."""
        for rank, chunk in enumerate(result_list, start=1):
            cid = getattr(chunk, "chunk_id", "")
            ccontent = getattr(chunk, "content", "")
            if chunk_id and cid == chunk_id:
                return rank
            if content and ccontent == content:
                return rank
        return 0
