"""Feedback data models for retrieval quality signals.

Provides :class:`FeedbackType` and :class:`RetrievalFeedback` for recording
user relevance signals (thumbs up/down, explicit ratings) on retrieved chunks.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class FeedbackType(Enum):
    """Categories of user feedback on a retrieval result."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    CLICK = "click"
    SKIP = "skip"


@dataclass
class RetrievalFeedback:
    """A single piece of user feedback on a retrieved chunk.

    Attributes:
        query: The query string that produced the retrieved result.
        chunk_id: Identifier of the chunk the feedback relates to.
        feedback_type: The kind of feedback signal.
        score: Normalised relevance score in ``[0.0, 1.0]``.
            For thumbs feedback: ``1.0`` = positive, ``0.0`` = negative.
            For ratings and clicks: caller-supplied value.
        id: Unique feedback event identifier.
        timestamp: UTC timestamp of the feedback event.
        metadata: Arbitrary extra information (e.g. session ID, rank shown).
    """

    query: str
    chunk_id: str
    feedback_type: FeedbackType
    score: float = 1.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def thumbs_up(
        cls,
        query: str,
        chunk_id: str,
        **metadata: Any,
    ) -> RetrievalFeedback:
        """Create a positive (thumbs-up) feedback event.

        Args:
            query: The query string.
            chunk_id: ID of the relevant chunk.
            **metadata: Extra metadata fields.

        Returns:
            A :class:`RetrievalFeedback` with score ``1.0``.
        """
        return cls(
            query=query,
            chunk_id=chunk_id,
            feedback_type=FeedbackType.THUMBS_UP,
            score=1.0,
            metadata=dict(metadata),
        )

    @classmethod
    def thumbs_down(
        cls,
        query: str,
        chunk_id: str,
        **metadata: Any,
    ) -> RetrievalFeedback:
        """Create a negative (thumbs-down) feedback event.

        Args:
            query: The query string.
            chunk_id: ID of the irrelevant chunk.
            **metadata: Extra metadata fields.

        Returns:
            A :class:`RetrievalFeedback` with score ``0.0``.
        """
        return cls(
            query=query,
            chunk_id=chunk_id,
            feedback_type=FeedbackType.THUMBS_DOWN,
            score=0.0,
            metadata=dict(metadata),
        )

    @classmethod
    def rated(
        cls,
        query: str,
        chunk_id: str,
        rating: float,
        **metadata: Any,
    ) -> RetrievalFeedback:
        """Create an explicit relevance rating feedback event.

        Args:
            query: The query string.
            chunk_id: ID of the rated chunk.
            rating: Relevance score in ``[0.0, 1.0]``.
            **metadata: Extra metadata fields.

        Returns:
            A :class:`RetrievalFeedback` with the clamped rating as score.

        Raises:
            ValueError: If ``rating`` is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= rating <= 1.0:
            raise ValueError(f"rating must be in [0.0, 1.0], got {rating!r}")
        return cls(
            query=query,
            chunk_id=chunk_id,
            feedback_type=FeedbackType.RATING,
            score=rating,
            metadata=dict(metadata),
        )

    @classmethod
    def clicked(
        cls,
        query: str,
        chunk_id: str,
        rank: int = 0,
        **metadata: Any,
    ) -> RetrievalFeedback:
        """Create a click-through feedback event.

        Args:
            query: The query string.
            chunk_id: ID of the clicked chunk.
            rank: The rank position at which the chunk was clicked (0-indexed).
            **metadata: Extra metadata.

        Returns:
            A :class:`RetrievalFeedback` with score derived from rank.
        """
        # Higher-rank clicks are treated as stronger signals
        score = 1.0 / (1.0 + rank) if rank >= 0 else 0.5
        return cls(
            query=query,
            chunk_id=chunk_id,
            feedback_type=FeedbackType.CLICK,
            score=score,
            metadata={"rank": rank, **metadata},
        )

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"RetrievalFeedback("
            f"type={self.feedback_type.value!r}, "
            f"score={self.score:.2f}, "
            f"chunk_id={self.chunk_id!r})"
        )
