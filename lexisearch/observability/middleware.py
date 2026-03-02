"""Observability middleware — automatic instrumentation for pipelines and APIs."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

from lexisearch.observability.base import SpanStatus
from lexisearch.observability.metrics import InMemoryMetricsCollector, LexiMetrics
from lexisearch.observability.tracing import InMemoryTracer

# ---------------------------------------------------------------------------
# Module-level singletons (can be replaced by the caller)
# ---------------------------------------------------------------------------


#: Global tracer — swap out for a real exporter in production.
_default_tracer: InMemoryTracer = InMemoryTracer()

#: Global metrics collector.
_default_metrics: InMemoryMetricsCollector = InMemoryMetricsCollector()


def get_tracer() -> InMemoryTracer:
    """Return the process-wide tracer instance.

    Returns:
        The active :class:`InMemoryTracer`.
    """
    return _default_tracer


def get_metrics() -> InMemoryMetricsCollector:
    """Return the process-wide metrics collector instance.

    Returns:
        The active :class:`InMemoryMetricsCollector`.
    """
    return _default_metrics


def set_tracer(tracer: InMemoryTracer) -> None:
    """Replace the global tracer.

    Args:
        tracer: New tracer to use for all instrumented calls.
    """
    global _default_tracer
    _default_tracer = tracer


def set_metrics(collector: InMemoryMetricsCollector) -> None:
    """Replace the global metrics collector.

    Args:
        collector: New collector to use for all instrumented calls.
    """
    global _default_metrics
    _default_metrics = collector


# ---------------------------------------------------------------------------
# Function decorator
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def instrument(
    operation: str | None = None,
    *,
    latency_metric: str | None = None,
    count_metric: str | None = None,
    error_metric: str = LexiMetrics.ERRORS_TOTAL,
) -> Callable[[F], F]:
    """Decorator that wraps a function with tracing and metrics.

    Automatically records:
    - A span in the global tracer.
    - Latency histogram (if *latency_metric* provided).
    - Call counter (if *count_metric* provided).
    - Error counter on exceptions.

    Args:
        operation: Span / operation name.  Defaults to ``func.__qualname__``.
        latency_metric: Histogram metric name for latency (ms).
        count_metric: Counter metric name incremented on each call.
        error_metric: Counter incremented on unhandled exceptions.

    Returns:
        Decorated function.

    Example:
        >>> @instrument("embed", latency_metric=LexiMetrics.EMBEDDING_LATENCY_MS)
        ... def embed_texts(texts: list[str]) -> list[list[float]]:
        ...     ...
    """

    def decorator(func: F) -> F:
        op_name = operation or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            metrics = get_metrics()
            span = tracer.start_span(op_name)
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                tracer.finish_span(span, status=SpanStatus.OK)
                return result
            except Exception:
                tracer.finish_span(span, status=SpanStatus.ERROR)
                metrics.increment(error_metric, operation=op_name)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1_000
                if latency_metric:
                    metrics.histogram(latency_metric, elapsed_ms, operation=op_name)
                if count_metric:
                    metrics.increment(count_metric, operation=op_name)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Context manager helper
# ---------------------------------------------------------------------------


class ObservabilityContext:
    """Context manager that captures tracing and metrics for a code block.

    Example:
        >>> with ObservabilityContext("retrieval") as ctx:
        ...     results = retriever.retrieve(query)
        ...     ctx.set_attribute("results_count", len(results))
    """

    def __init__(
        self,
        operation: str,
        *,
        latency_metric: str | None = None,
        tracer: InMemoryTracer | None = None,
        metrics: InMemoryMetricsCollector | None = None,
    ) -> None:
        """Initialise the context.

        Args:
            operation: Span name.
            latency_metric: If provided, record latency (ms) in this histogram.
            tracer: Override the global tracer.
            metrics: Override the global metrics collector.
        """
        self.operation = operation
        self.latency_metric = latency_metric
        self._tracer = tracer or get_tracer()
        self._metrics = metrics or get_metrics()
        self._span: Any = None
        self._t0: float = 0.0

    def __enter__(self) -> ObservabilityContext:
        """Start the span and timer."""
        self._span = self._tracer.start_span(self.operation)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Finish the span, recording latency and any errors."""
        elapsed_ms = (time.perf_counter() - self._t0) * 1_000
        status = SpanStatus.ERROR if exc_type else SpanStatus.OK
        self._tracer.finish_span(self._span, status=status)
        if self.latency_metric:
            self._metrics.histogram(self.latency_metric, elapsed_ms)
        if exc_type:
            self._metrics.increment(LexiMetrics.ERRORS_TOTAL, operation=self.operation)

    def set_attribute(self, key: str, value: Any) -> None:
        """Add a key-value attribute to the current span.

        Args:
            key: Attribute name.
            value: Attribute value.
        """
        if self._span is not None:
            self._span.set_attribute(key, value)

    def add_event(self, name: str, **attrs: Any) -> None:
        """Attach a named event to the current span.

        Args:
            name: Event name.
            **attrs: Event attributes.
        """
        if self._span is not None:
            self._span.add_event(name, **attrs)
