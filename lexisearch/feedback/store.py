"""Feedback storage backends.

Provides :class:`InMemoryFeedbackStore` for ephemeral storage and
:class:`DiskFeedbackStore` for JSON-file-backed persistence.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from lexisearch.feedback.models import FeedbackType, RetrievalFeedback


class InMemoryFeedbackStore:
    """In-memory storage for retrieval feedback events.

    Stores all recorded :class:`~lexisearch.feedback.models.RetrievalFeedback`
    objects in a list and maintains an index keyed by ``(query, chunk_id)``
    for efficient lookup.

    Example:
        >>> store = InMemoryFeedbackStore()
        >>> store.record(RetrievalFeedback.thumbs_up("query", "chunk-1"))
        >>> store.aggregate_chunk_score("chunk-1")
        1.0
    """

    def __init__(self) -> None:
        """Initialise an empty in-memory store."""
        self._feedback: list[RetrievalFeedback] = []
        self._index: dict[tuple[str, str], list[RetrievalFeedback]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, feedback: RetrievalFeedback) -> None:
        """Record a feedback event.

        Args:
            feedback: The :class:`RetrievalFeedback` to store.
        """
        self._feedback.append(feedback)
        self._index[(feedback.query, feedback.chunk_id)].append(feedback)

    def record_many(self, feedbacks: list[RetrievalFeedback]) -> None:
        """Record multiple feedback events at once.

        Args:
            feedbacks: List of feedback objects to store.
        """
        for fb in feedbacks:
            self.record(fb)

    def clear(self) -> None:
        """Remove all stored feedback."""
        self._feedback.clear()
        self._index.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_for_chunk(self, chunk_id: str) -> list[RetrievalFeedback]:
        """Return all feedback events for a given chunk (any query).

        Args:
            chunk_id: The chunk identifier to look up.

        Returns:
            All feedback events referencing this chunk.
        """
        return [f for f in self._feedback if f.chunk_id == chunk_id]

    def get_for_query(self, query: str) -> list[RetrievalFeedback]:
        """Return all feedback events for a given query (any chunk).

        Args:
            query: The query string to look up.

        Returns:
            All feedback events for this query.
        """
        return [f for f in self._feedback if f.query == query]

    def get_for_query_and_chunk(self, query: str, chunk_id: str) -> list[RetrievalFeedback]:
        """Return feedback events for a specific (query, chunk) pair.

        Args:
            query: The query string.
            chunk_id: The chunk identifier.

        Returns:
            Feedback events matching both query and chunk.
        """
        return self._index.get((query, chunk_id), [])

    def aggregate_chunk_score(self, chunk_id: str) -> float | None:
        """Compute the mean relevance score for a chunk across all queries.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            Mean score in ``[0.0, 1.0]``, or ``None`` if no feedback exists.
        """
        fb = self.get_for_chunk(chunk_id)
        if not fb:
            return None
        return sum(f.score for f in fb) / len(fb)

    def aggregate_query_chunk_score(self, query: str, chunk_id: str) -> float | None:
        """Compute mean score for a specific (query, chunk) pair.

        Args:
            query: The query string.
            chunk_id: The chunk identifier.

        Returns:
            Mean score or ``None``.
        """
        fb = self.get_for_query_and_chunk(query, chunk_id)
        if not fb:
            return None
        return sum(f.score for f in fb) / len(fb)

    def total_feedback(self) -> int:
        """Return total number of recorded feedback events.

        Returns:
            Count of all stored feedback events.
        """
        return len(self._feedback)

    def positive_rate(self, chunk_id: str) -> float | None:
        """Compute the fraction of positive feedback events for a chunk.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            Fraction of events with score >= 0.5, or ``None`` if no feedback.
        """
        fb = self.get_for_chunk(chunk_id)
        if not fb:
            return None
        positives = sum(1 for f in fb if f.score >= 0.5)
        return positives / len(fb)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialise all feedback to a list of plain dicts.

        Returns:
            JSON-serialisable representation of all feedback.
        """
        return [
            {
                "id": f.id,
                "query": f.query,
                "chunk_id": f.chunk_id,
                "feedback_type": f.feedback_type.value,
                "score": f.score,
                "timestamp": f.timestamp.isoformat(),
                "metadata": f.metadata,
            }
            for f in self._feedback
        ]

    @classmethod
    def from_dicts(cls, data: list[dict[str, Any]]) -> InMemoryFeedbackStore:
        """Create a store from a list of serialised dicts.

        Args:
            data: Output of :meth:`to_dicts`.

        Returns:
            Populated :class:`InMemoryFeedbackStore`.
        """
        store = cls()
        for item in data:
            fb = RetrievalFeedback(
                query=item["query"],
                chunk_id=item["chunk_id"],
                feedback_type=FeedbackType(item["feedback_type"]),
                score=item["score"],
                id=item["id"],
                timestamp=datetime.fromisoformat(item["timestamp"]),
                metadata=item.get("metadata", {}),
            )
            store.record(fb)
        return store

    def __len__(self) -> int:
        """Return total feedback count."""
        return len(self._feedback)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"InMemoryFeedbackStore(events={len(self)})"


class DiskFeedbackStore(InMemoryFeedbackStore):
    """JSON-file-backed feedback store.

    Extends :class:`InMemoryFeedbackStore` with automatic persistence to a
    JSON file.  All feedback is kept in memory for fast querying; writes are
    serialised to disk on every :meth:`record` call.

    Args:
        path: Path to the JSON file.  Created on first write if absent.

    Example:
        >>> store = DiskFeedbackStore("feedback.json")
        >>> store.record(RetrievalFeedback.thumbs_up("query", "chunk-1"))
        >>> store2 = DiskFeedbackStore("feedback.json")  # reloaded from disk
        >>> store2.total_feedback()
        1
    """

    def __init__(self, path: str | Path) -> None:
        """Initialise and load any existing feedback from disk.

        Args:
            path: File path for persisted JSON data.
        """
        super().__init__()
        self._path = Path(path)
        self._load()

    def record(self, feedback: RetrievalFeedback) -> None:
        """Record a feedback event and persist to disk.

        Args:
            feedback: The :class:`RetrievalFeedback` to store.
        """
        super().record(feedback)
        self._save()

    def record_many(self, feedbacks: list[RetrievalFeedback]) -> None:
        """Record multiple events and persist once.

        Args:
            feedbacks: List of feedback objects.
        """
        for fb in feedbacks:
            super().record(fb)
        self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self.to_dicts(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            store = InMemoryFeedbackStore.from_dicts(data)
            self._feedback = store._feedback
            self._index = store._index
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # Corrupt file — start fresh

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"DiskFeedbackStore(path={str(self._path)!r}, events={len(self)})"
