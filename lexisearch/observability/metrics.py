"""In-process metrics collection — counters, gauges, and histograms."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from lexisearch.observability.base import BaseMetricsCollector, MetricKind, MetricPoint

# ---------------------------------------------------------------------------
# Histogram summary
# ---------------------------------------------------------------------------


@dataclass
class HistogramSummary:
    """Aggregated statistics for a histogram metric.

    Attributes:
        name: Metric name.
        count: Number of observations.
        total: Sum of all observed values.
        min: Minimum observed value.
        max: Maximum observed value.
        mean: Arithmetic mean of observations.
        p50: 50th percentile (median).
        p95: 95th percentile.
        p99: 99th percentile.
    """

    name: str
    count: int
    total: float
    min: float
    max: float
    mean: float
    p50: float
    p95: float
    p99: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise summary to a plain dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "total": self.total,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


# ---------------------------------------------------------------------------
# In-memory collector
# ---------------------------------------------------------------------------


class InMemoryMetricsCollector(BaseMetricsCollector):
    """Collects metrics in memory.

    Suitable for testing and local development.  Provides aggregation helpers
    for counters, gauges, and histograms.

    Attributes:
        max_points: Maximum raw metric points to retain.
    """

    def __init__(self, max_points: int = 100_000) -> None:
        """Initialise the in-memory metrics collector.

        Args:
            max_points: Maximum raw points before FIFO eviction.
        """
        self.max_points = max_points
        self._points: list[MetricPoint] = []
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histogram_values: dict[str, list[float]] = defaultdict(list)

    def _make_label_key(self, name: str, labels: dict[str, str]) -> str:
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}" if label_str else name

    def _store(self, point: MetricPoint) -> None:
        if len(self._points) >= self.max_points:
            self._points.pop(0)
        self._points.append(point)

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Increment a counter by *value*.

        Args:
            name: Counter metric name.
            value: Amount to add (must be positive).
            **labels: Arbitrary label key-value pairs.
        """
        label_key = self._make_label_key(name, labels)
        self._counters[label_key] += value
        self._store(MetricPoint(name=name, kind=MetricKind.COUNTER, value=value, labels=labels))

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a gauge to an absolute *value*.

        Args:
            name: Gauge metric name.
            value: Current absolute value.
            **labels: Arbitrary label key-value pairs.
        """
        label_key = self._make_label_key(name, labels)
        self._gauges[label_key] = value
        self._store(MetricPoint(name=name, kind=MetricKind.GAUGE, value=value, labels=labels))

    def histogram(self, name: str, value: float, **labels: str) -> None:
        """Record an observed *value* in a histogram.

        Args:
            name: Histogram metric name.
            value: Observed value (e.g. latency in ms, token count).
            **labels: Arbitrary label key-value pairs.
        """
        label_key = self._make_label_key(name, labels)
        self._histogram_values[label_key].append(value)
        self._store(MetricPoint(name=name, kind=MetricKind.HISTOGRAM, value=value, labels=labels))

    def get_metrics(self) -> list[MetricPoint]:
        """Return all raw metric points (newest last).

        Returns:
            List of :class:`MetricPoint` objects.
        """
        return list(self._points)

    def get_counter(self, name: str, **labels: str) -> float:
        """Return the current total for a counter.

        Args:
            name: Counter name.
            **labels: Label filters.

        Returns:
            Accumulated counter value (0.0 if not found).
        """
        return self._counters.get(self._make_label_key(name, labels), 0.0)

    def get_gauge(self, name: str, **labels: str) -> float | None:
        """Return the last recorded gauge value.

        Args:
            name: Gauge name.
            **labels: Label filters.

        Returns:
            Last recorded value, or ``None`` if never set.
        """
        return self._gauges.get(self._make_label_key(name, labels))

    def summarize_histogram(self, name: str, **labels: str) -> HistogramSummary | None:
        """Compute summary statistics for a histogram.

        Args:
            name: Histogram name.
            **labels: Label filters.

        Returns:
            :class:`HistogramSummary`, or ``None`` if no data.
        """
        label_key = self._make_label_key(name, labels)
        values = self._histogram_values.get(label_key)
        if not values:
            return None
        sorted_vals = sorted(values)

        def percentile(p: float) -> float:
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

        return HistogramSummary(
            name=name,
            count=len(sorted_vals),
            total=sum(sorted_vals),
            min=sorted_vals[0],
            max=sorted_vals[-1],
            mean=statistics.mean(sorted_vals),
            p50=percentile(50),
            p95=percentile(95),
            p99=percentile(99),
        )

    def clear(self) -> None:
        """Reset all collected metrics."""
        self._points.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histogram_values.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return a human-readable snapshot of current metric state.

        Returns:
            Dictionary with ``counters``, ``gauges``, and ``histograms`` keys.
        """
        histograms: dict[str, Any] = {}
        for key in self._histogram_values:
            name, _, _ = key.partition("{")
            summary = self.summarize_histogram(name)
            if summary:
                histograms[key] = summary.to_dict()
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": histograms,
        }


# ---------------------------------------------------------------------------
# Pre-defined metric names (constants)
# ---------------------------------------------------------------------------


class LexiMetrics:
    """Canonical metric names used throughout LexiSearch.

    Use these constants instead of raw strings to avoid typos and to enable
    IDE auto-complete.
    """

    # Query pipeline
    QUERIES_TOTAL = "lexisearch.queries.total"
    QUERY_LATENCY_MS = "lexisearch.query.latency_ms"
    RETRIEVAL_LATENCY_MS = "lexisearch.retrieval.latency_ms"
    GENERATION_LATENCY_MS = "lexisearch.generation.latency_ms"

    # Ingest pipeline
    DOCUMENTS_INGESTED = "lexisearch.documents.ingested"
    CHUNKS_CREATED = "lexisearch.chunks.created"
    INGEST_LATENCY_MS = "lexisearch.ingest.latency_ms"

    # Embedding
    EMBEDDINGS_CREATED = "lexisearch.embeddings.created"
    EMBEDDING_LATENCY_MS = "lexisearch.embedding.latency_ms"

    # Vector store
    VECTORSTORE_SIZE = "lexisearch.vectorstore.size"
    VECTORSTORE_SEARCH_LATENCY_MS = "lexisearch.vectorstore.search_latency_ms"

    # Cache
    CACHE_HITS = "lexisearch.cache.hits"
    CACHE_MISSES = "lexisearch.cache.misses"

    # LLM tokens / cost
    TOKENS_PROMPT = "lexisearch.tokens.prompt"
    TOKENS_COMPLETION = "lexisearch.tokens.completion"
    TOKENS_TOTAL = "lexisearch.tokens.total"
    ESTIMATED_COST_USD = "lexisearch.cost.estimated_usd"

    # Errors
    ERRORS_TOTAL = "lexisearch.errors.total"

    @staticmethod
    def all_names() -> list[str]:
        """Return all canonical metric name values.

        Returns:
            Sorted list of metric name strings.
        """
        return sorted(
            v for k, v in vars(LexiMetrics).items() if not k.startswith("_") and isinstance(v, str)
        )
