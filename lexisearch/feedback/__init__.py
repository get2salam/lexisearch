"""Relevance feedback loop for LexiSearch RAG pipelines.

This package lets applications collect explicit or implicit user feedback on
retrieved chunks and use that signal to improve future retrieval rankings.

Components:

- :class:`~lexisearch.feedback.models.FeedbackType` - thumbs up/down, rating, click.
- :class:`~lexisearch.feedback.models.RetrievalFeedback` - a single feedback event.
- :class:`~lexisearch.feedback.store.InMemoryFeedbackStore` - ephemeral feedback store.
- :class:`~lexisearch.feedback.store.DiskFeedbackStore` - JSON-file-backed store.
- :class:`~lexisearch.feedback.ranker.FeedbackRanker` - score-adjustment re-ranker.

Example:
    >>> from lexisearch.feedback import (
    ...     FeedbackRanker,
    ...     InMemoryFeedbackStore,
    ...     RetrievalFeedback,
    ... )
    >>> store = InMemoryFeedbackStore()
    >>> store.record(RetrievalFeedback.thumbs_up("What is RAG?", "chunk-abc"))
    >>> ranker = FeedbackRanker(store)
    >>> reranked = ranker.rerank(results, query="What is RAG?")
"""

from __future__ import annotations

from lexisearch.feedback.models import FeedbackType, RetrievalFeedback
from lexisearch.feedback.ranker import FeedbackRanker
from lexisearch.feedback.store import DiskFeedbackStore, InMemoryFeedbackStore

__all__ = [
    "DiskFeedbackStore",
    "FeedbackRanker",
    "FeedbackType",
    "InMemoryFeedbackStore",
    "RetrievalFeedback",
]
