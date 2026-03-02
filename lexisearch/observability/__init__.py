"""Observability layer — tracing, metrics, structured logging, and cost tracking.

Provides zero-dependency, in-process instrumentation that is compatible with
OpenTelemetry conventions.  Swap out the default in-memory collectors for
real exporters (OTLP, Prometheus, Jaeger) in production.

Quick start::

    from lexisearch.observability import (
        InMemoryTracer,
        InMemoryMetricsCollector,
        CostTracker,
        instrument,
        LexiMetrics,
        ObservabilityContext,
        get_tracer,
        get_metrics,
    )

    # Trace a block
    tracer = InMemoryTracer()
    with tracer.trace("my_operation") as span:
        span.set_attribute("doc_count", 10)

    # Collect metrics
    metrics = InMemoryMetricsCollector()
    metrics.increment(LexiMetrics.QUERIES_TOTAL)
    metrics.histogram(LexiMetrics.QUERY_LATENCY_MS, 42.3)

    # Track LLM cost
    tracker = CostTracker()
    tracker.record("gpt-4o-mini", prompt_tokens=500, completion_tokens=100)
    print(tracker.summary())
"""

from __future__ import annotations

from lexisearch.observability.base import (
    BaseMetricsCollector,
    BaseTracer,
    MetricKind,
    MetricPoint,
    Span,
    SpanEvent,
    SpanStatus,
)
from lexisearch.observability.cost_tracker import (
    DEFAULT_PRICING,
    CostTracker,
    TokenUsage,
)
from lexisearch.observability.metrics import (
    HistogramSummary,
    InMemoryMetricsCollector,
    LexiMetrics,
)
from lexisearch.observability.middleware import (
    ObservabilityContext,
    get_metrics,
    get_tracer,
    instrument,
    set_metrics,
    set_tracer,
)
from lexisearch.observability.tracing import InMemoryTracer, NoOpTracer

__all__ = [
    "DEFAULT_PRICING",
    # Base
    "BaseMetricsCollector",
    "BaseTracer",
    # Cost tracking
    "CostTracker",
    # Metrics
    "HistogramSummary",
    "InMemoryMetricsCollector",
    # Tracing
    "InMemoryTracer",
    "LexiMetrics",
    "MetricKind",
    "MetricPoint",
    "NoOpTracer",
    # Middleware
    "ObservabilityContext",
    "Span",
    "SpanEvent",
    "SpanStatus",
    "TokenUsage",
    "get_metrics",
    "get_tracer",
    "instrument",
    "set_metrics",
    "set_tracer",
]
