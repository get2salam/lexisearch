"""Abstract base class for retrieval backends.

Every retriever in LexiSearch implements this interface, enabling seamless
composition of sparse, dense, and hybrid retrieval strategies.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lexisearch.models import Chunk, SearchResponse, SearchResult


class RetrieverType(Enum):
    """Supported retriever backend types."""

    SPARSE = "sparse"
    DENSE = "dense"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class FilterOperator(Enum):
    """Operators for metadata filtering."""

    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


@dataclass(frozen=True)
class MetadataFilter:
    """A single metadata filter condition.

    Attributes:
        field: The metadata field name to filter on.
        operator: The comparison operator.
        value: The value to compare against.
    """

    field: str
    operator: FilterOperator
    value: Any


@dataclass
class RetrieverConfig:
    """Shared configuration for retriever initialisation.

    Attributes:
        top_k: Default number of results to return.
        score_threshold: Minimum score threshold (results below are discarded).
        filters: Default metadata filters applied to every query.
        extra: Backend-specific options.
    """

    top_k: int = 10
    score_threshold: float = 0.0
    filters: list[MetadataFilter] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract retriever with query and scoring operations.

    Implementations must override :meth:`retrieve` and :meth:`retriever_type`.
    The base class provides convenience wrappers for filtering, scoring
    thresholds, and timed search responses.

    Args:
        config: A :class:`RetrieverConfig` describing retriever behaviour.
    """

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        """Initialize provides."""
        self.config = config or RetrieverConfig()

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: list[MetadataFilter] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Retrieve the most relevant chunks for the given query.

        Args:
            query: The natural-language query string.
            top_k: Maximum number of results (overrides config default).
            filters: Additional metadata filters for this query.
            **kwargs: Backend-specific parameters.

        Returns:
            Ordered list of :class:`SearchResult` (best first).
        """
        ...

    @abstractmethod
    def retriever_type(self) -> RetrieverType:
        """Return the type of this retriever.

        Returns:
            The :class:`RetrieverType` enum value.
        """
        ...

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: list[MetadataFilter] | None = None,
        **kwargs: Any,
    ) -> SearchResponse:
        """Execute a timed retrieval and return a full :class:`SearchResponse`.

        This wraps :meth:`retrieve` with timing, score thresholds, and
        rank assignment.

        Args:
            query: The natural-language query string.
            top_k: Maximum number of results.
            filters: Metadata filters.
            **kwargs: Passed to :meth:`retrieve`.

        Returns:
            A :class:`SearchResponse` with ranked results and latency.
        """
        k = top_k or self.config.top_k
        merged_filters = list(self.config.filters)
        if filters:
            merged_filters.extend(filters)

        start = time.perf_counter()
        results = self.retrieve(
            query,
            top_k=k,
            filters=merged_filters if merged_filters else None,
            **kwargs,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Apply score threshold
        if self.config.score_threshold > 0:
            results = [r for r in results if r.score >= self.config.score_threshold]

        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            latency_ms=round(elapsed_ms, 2),
        )

    def apply_metadata_filter(self, chunk: Chunk, filters: list[MetadataFilter]) -> bool:
        """Check if a chunk passes all metadata filters.

        Args:
            chunk: The chunk to test.
            filters: Filters to apply.

        Returns:
            ``True`` if the chunk passes all filters.
        """
        for f in filters:
            value = chunk.metadata.get(f.field)
            if not self._check_filter(value, f):
                return False
        return True

    @staticmethod
    def _check_filter(value: Any, meta_filter: MetadataFilter) -> bool:
        """Evaluate a single filter condition against a value.

        Args:
            value: The actual metadata value (may be ``None``).
            meta_filter: The filter to check.

        Returns:
            ``True`` if the condition is satisfied.
        """
        op = meta_filter.operator
        target = meta_filter.value

        if value is None:
            return op in (FilterOperator.NEQ, FilterOperator.NOT_IN)

        if op is FilterOperator.EQ:
            return bool(value == target)
        if op is FilterOperator.NEQ:
            return bool(value != target)
        if op is FilterOperator.GT:
            return bool(value > target)
        if op is FilterOperator.GTE:
            return bool(value >= target)
        if op is FilterOperator.LT:
            return bool(value < target)
        if op is FilterOperator.LTE:
            return bool(value <= target)
        if op is FilterOperator.IN:
            return bool(value in target)
        if op is FilterOperator.NOT_IN:
            return bool(value not in target)
        if op is FilterOperator.CONTAINS:
            return bool(target in value)
        return False

    def get_config(self) -> dict[str, Any]:
        """Return the retriever configuration as a serialisable dictionary.

        Returns:
            Configuration dict.
        """
        return {
            "type": self.retriever_type().value,
            "top_k": self.config.top_k,
            "score_threshold": self.config.score_threshold,
            "num_filters": len(self.config.filters),
            "extra": self.config.extra,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{type(self).__name__}("
            f"type={self.retriever_type().value!r}, "
            f"top_k={self.config.top_k})"
        )
