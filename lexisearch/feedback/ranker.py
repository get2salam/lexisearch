"""Feedback-informed re-ranker.

:class:`FeedbackRanker` adjusts retrieval scores based on historical
user feedback, boosting highly-rated chunks and penalising negatively-
rated ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lexisearch.models import SearchResult

if TYPE_CHECKING:
    from lexisearch.feedback.store import InMemoryFeedbackStore


class FeedbackRanker:
    """Re-ranks search results using historical relevance feedback.

    For each :class:`~lexisearch.models.SearchResult`, the ranker looks up
    the aggregate feedback score for the chunk and applies a linear boost or
    penalty to the retrieval score:

    - Chunks with ``aggregate_score >= threshold_positive`` receive a boost.
    - Chunks with ``aggregate_score <= threshold_negative`` receive a penalty.
    - Chunks with no feedback are left unchanged.

    The adjusted scores are re-sorted and re-ranked in descending order.

    Args:
        store: Feedback store to query for historical signals.
        boost: Score delta added for positively-rated chunks (default 0.2).
        penalty: Score delta subtracted for negatively-rated chunks (default 0.3).
        threshold_positive: Minimum aggregate score to trigger a boost (default 0.6).
        threshold_negative: Maximum aggregate score to trigger a penalty (default 0.4).
        query_aware: When ``True``, use per-query feedback scores in addition
            to global chunk scores.

    Example:
        >>> store = InMemoryFeedbackStore()
        >>> store.record(RetrievalFeedback.thumbs_up("What is RAG?", "chunk-1"))
        >>> ranker = FeedbackRanker(store)
        >>> results = ranker.rerank(original_results)
    """

    def __init__(
        self,
        store: InMemoryFeedbackStore,
        boost: float = 0.2,
        penalty: float = 0.3,
        threshold_positive: float = 0.6,
        threshold_negative: float = 0.4,
        query_aware: bool = True,
    ) -> None:
        """Initialise the FeedbackRanker.

        Args:
            store: Feedback data source.
            boost: Score increase for positively-rated chunks.
            penalty: Score decrease for negatively-rated chunks.
            threshold_positive: Aggregate score threshold for boost.
            threshold_negative: Aggregate score threshold for penalty.
            query_aware: Prefer per-query scores when available.
        """
        self.store = store
        self.boost = boost
        self.penalty = penalty
        self.threshold_positive = threshold_positive
        self.threshold_negative = threshold_negative
        self.query_aware = query_aware

    def rerank(
        self,
        results: list[SearchResult],
        query: str | None = None,
    ) -> list[SearchResult]:
        """Apply feedback-informed score adjustments and re-sort.

        Args:
            results: Original retrieval results to re-rank.
            query: The current query string.  When provided and
                ``query_aware=True``, per-query scores override global ones.

        Returns:
            Re-ranked list of :class:`~lexisearch.models.SearchResult` with
            adjusted scores and updated ranks.
        """
        adjusted: list[SearchResult] = []
        for result in results:
            fb_score = self._get_score(result.chunk.id, query)
            new_score = self._adjust(result.score, fb_score)
            adjusted.append(
                SearchResult(
                    chunk=result.chunk,
                    score=new_score,
                    rank=result.rank,
                    metadata={
                        **result.metadata,
                        "feedback_adjusted": fb_score is not None,
                        "feedback_score": fb_score,
                    },
                )
            )

        adjusted.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(adjusted):
            r.rank = i + 1

        return adjusted

    def _get_score(self, chunk_id: str, query: str | None) -> float | None:
        """Resolve the applicable feedback score for a chunk.

        Args:
            chunk_id: The chunk to look up.
            query: Optional current query for per-query lookup.

        Returns:
            Feedback score or ``None`` if no data exists.
        """
        if self.query_aware and query:
            per_query = self.store.aggregate_query_chunk_score(query, chunk_id)
            if per_query is not None:
                return per_query
        return self.store.aggregate_chunk_score(chunk_id)

    def _adjust(self, score: float, fb_score: float | None) -> float:
        """Compute the adjusted retrieval score.

        Args:
            score: Original retrieval score.
            fb_score: Aggregate feedback score, or ``None``.

        Returns:
            Adjusted score clamped to ``[0.0, 1.0]``.
        """
        if fb_score is None:
            return score
        if fb_score >= self.threshold_positive:
            return min(1.0, score + self.boost)
        if fb_score <= self.threshold_negative:
            return max(0.0, score - self.penalty)
        return score
