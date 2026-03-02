"""Tests for the observability layer."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from lexisearch.observability.base import MetricKind, SpanStatus
from lexisearch.observability.cost_tracker import CostTracker, TokenUsage
from lexisearch.observability.metrics import InMemoryMetricsCollector, LexiMetrics
from lexisearch.observability.middleware import (
    ObservabilityContext,
    get_metrics,
    get_tracer,
    instrument,
    set_metrics,
    set_tracer,
)
from lexisearch.observability.tracing import InMemoryTracer, NoOpTracer

# ---------------------------------------------------------------------------
# Tracing tests
# ---------------------------------------------------------------------------


class TestNoOpTracer:
    def test_start_span_returns_span(self):
        tracer = NoOpTracer()
        span = tracer.start_span("test")
        assert span.name == "test"

    def test_finish_span_does_not_raise(self):
        tracer = NoOpTracer()
        span = tracer.start_span("test")
        tracer.finish_span(span)

    def test_get_spans_returns_empty(self):
        tracer = NoOpTracer()
        assert tracer.get_spans() == []


class TestInMemoryTracer:
    def test_start_and_finish_span(self):
        tracer = InMemoryTracer()
        span = tracer.start_span("operation.test", attributes={"key": "value"})
        assert span.name == "operation.test"
        assert span.attributes["key"] == "value"
        assert span.end_time is None

        tracer.finish_span(span)
        assert span.end_time is not None
        assert span.status == SpanStatus.OK

    def test_finish_with_error_status(self):
        tracer = InMemoryTracer()
        span = tracer.start_span("failing_op")
        tracer.finish_span(span, status=SpanStatus.ERROR)
        assert span.status == SpanStatus.ERROR

    def test_span_duration_ms(self):
        tracer = InMemoryTracer()
        span = tracer.start_span("timed_op")
        time.sleep(0.01)
        tracer.finish_span(span)
        assert span.duration_ms is not None
        assert span.duration_ms >= 5

    def test_get_spans(self):
        tracer = InMemoryTracer()
        for i in range(3):
            s = tracer.start_span(f"op_{i}")
            tracer.finish_span(s)
        assert len(tracer.get_spans()) == 3

    def test_get_spans_by_trace_id(self):
        tracer = InMemoryTracer()
        span = tracer.start_span("first")
        tracer.finish_span(span)
        trace_spans = tracer.get_spans(trace_id=span.trace_id)
        assert len(trace_spans) == 1
        assert trace_spans[0].trace_id == span.trace_id

    def test_clear(self):
        tracer = InMemoryTracer()
        s = tracer.start_span("op")
        tracer.finish_span(s)
        tracer.clear()
        assert tracer.span_count == 0

    def test_max_spans_eviction(self):
        tracer = InMemoryTracer(max_spans=5)
        for i in range(10):
            s = tracer.start_span(f"op_{i}")
            tracer.finish_span(s)
        assert tracer.span_count <= 5

    def test_context_manager_ok(self):
        tracer = InMemoryTracer()
        with tracer.trace("ctx_op") as span:
            span.set_attribute("k", "v")
        assert span.status == SpanStatus.OK
        assert span.attributes["k"] == "v"

    def test_context_manager_error(self):
        tracer = InMemoryTracer()
        with pytest.raises(ValueError), tracer.trace("failing") as span:
            raise ValueError("boom")
        assert span.status == SpanStatus.ERROR

    def test_span_add_event(self):
        tracer = InMemoryTracer()
        with tracer.trace("evt_op") as span:
            span.add_event("checkpoint", count=42)
        assert len(span.events) == 1
        assert span.events[0].name == "checkpoint"
        assert span.events[0].attributes["count"] == 42

    def test_span_to_dict(self):
        tracer = InMemoryTracer()
        s = tracer.start_span("dict_op")
        tracer.finish_span(s)
        d = s.to_dict()
        assert d["name"] == "dict_op"
        assert "duration_ms" in d
        assert d["status"] == "ok"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestInMemoryMetricsCollector:
    def test_increment_counter(self):
        m = InMemoryMetricsCollector()
        m.increment("test.counter")
        assert m.get_counter("test.counter") == 1.0

    def test_increment_by_value(self):
        m = InMemoryMetricsCollector()
        m.increment("test.counter", 5.0)
        m.increment("test.counter", 3.0)
        assert m.get_counter("test.counter") == 8.0

    def test_increment_with_labels(self):
        m = InMemoryMetricsCollector()
        m.increment("requests", operation="query")
        m.increment("requests", operation="ingest")
        assert m.get_counter("requests", operation="query") == 1.0
        assert m.get_counter("requests", operation="ingest") == 1.0

    def test_gauge(self):
        m = InMemoryMetricsCollector()
        m.gauge("store.size", 100.0)
        assert m.get_gauge("store.size") == 100.0
        m.gauge("store.size", 200.0)
        assert m.get_gauge("store.size") == 200.0

    def test_histogram(self):
        m = InMemoryMetricsCollector()
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            m.histogram("latency", v)
        summary = m.summarize_histogram("latency")
        assert summary is not None
        assert summary.count == 5
        assert summary.min == 10.0
        assert summary.max == 50.0
        assert summary.mean == 30.0
        assert summary.p50 == 30.0

    def test_histogram_none_when_empty(self):
        m = InMemoryMetricsCollector()
        assert m.summarize_histogram("nonexistent") is None

    def test_get_metrics_returns_points(self):
        m = InMemoryMetricsCollector()
        m.increment("a")
        m.gauge("b", 1.0)
        m.histogram("c", 2.0)
        points = m.get_metrics()
        assert len(points) == 3

    def test_metric_kinds(self):
        m = InMemoryMetricsCollector()
        m.increment("cnt")
        m.gauge("g", 1.0)
        m.histogram("h", 5.0)
        kinds = {p.kind for p in m.get_metrics()}
        assert MetricKind.COUNTER in kinds
        assert MetricKind.GAUGE in kinds
        assert MetricKind.HISTOGRAM in kinds

    def test_clear(self):
        m = InMemoryMetricsCollector()
        m.increment("x")
        m.clear()
        assert m.get_metrics() == []

    def test_snapshot(self):
        m = InMemoryMetricsCollector()
        m.increment("cnt", 3.0)
        m.gauge("size", 42.0)
        m.histogram("lat", 10.0)
        snap = m.snapshot()
        assert "counters" in snap
        assert "gauges" in snap
        assert "histograms" in snap

    def test_max_points_eviction(self):
        m = InMemoryMetricsCollector(max_points=5)
        for i in range(10):
            m.increment(f"m_{i}")
        assert len(m.get_metrics()) <= 5

    def test_histogram_summary_to_dict(self):
        m = InMemoryMetricsCollector()
        for v in [1.0, 2.0, 3.0]:
            m.histogram("x", v)
        summary = m.summarize_histogram("x")
        assert summary is not None
        d = summary.to_dict()
        assert d["count"] == 3


class TestLexiMetrics:
    def test_all_names_returns_list(self):
        names = LexiMetrics.all_names()
        assert len(names) > 5
        assert all(isinstance(n, str) for n in names)
        assert LexiMetrics.QUERIES_TOTAL in names

    def test_metric_name_format(self):
        for name in LexiMetrics.all_names():
            assert name.startswith("lexisearch.")


# ---------------------------------------------------------------------------
# Cost tracker tests
# ---------------------------------------------------------------------------


class TestCostTracker:
    def test_record_known_model(self):
        tracker = CostTracker()
        usage = tracker.record("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 1000
        assert usage.completion_tokens == 500
        assert usage.total_tokens == 1500
        assert usage.estimated_cost_usd > 0

    def test_record_unknown_model(self):
        tracker = CostTracker()
        usage = tracker.record("unknown-model", prompt_tokens=100, completion_tokens=50)
        assert usage.estimated_cost_usd == 0.0

    def test_total_tokens(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
        tracker.record("gpt-4o-mini", prompt_tokens=200, completion_tokens=100)
        assert tracker.total_prompt_tokens == 300
        assert tracker.total_completion_tokens == 150
        assert tracker.total_tokens == 450

    def test_total_cost(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", prompt_tokens=1000, completion_tokens=1000)
        assert tracker.total_cost_usd > 0

    def test_by_model(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
        tracker.record("gpt-4o", prompt_tokens=50, completion_tokens=25)
        by_model = tracker.by_model()
        assert "gpt-4o-mini" in by_model
        assert "gpt-4o" in by_model
        assert by_model["gpt-4o-mini"]["calls"] == 1

    def test_summary(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", prompt_tokens=500, completion_tokens=100)
        summary = tracker.summary()
        assert summary["total_calls"] == 1
        assert summary["total_tokens"] == 600
        assert "by_model" in summary

    def test_reset(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
        tracker.reset()
        assert tracker.total_tokens == 0

    def test_estimate_cost_without_recording(self):
        tracker = CostTracker()
        cost = tracker.estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=1000)
        assert cost > 0

    def test_custom_pricing(self):
        custom = {"my-model": (0.01, 0.02)}
        tracker = CostTracker(pricing=custom)
        cost = tracker.estimate_cost("my-model", prompt_tokens=1000, completion_tokens=1000)
        assert abs(cost - 0.03) < 1e-6

    def test_metadata_stored(self):
        tracker = CostTracker()
        usage = tracker.record(
            "gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
            operation="query",
        )
        assert usage.metadata["operation"] == "query"


# ---------------------------------------------------------------------------
# Middleware tests
# ---------------------------------------------------------------------------


class TestInstrumentDecorator:
    def test_wraps_function(self):
        m = InMemoryMetricsCollector()
        t = InMemoryTracer()

        @instrument("test_op", latency_metric="test.latency", count_metric="test.count")
        def my_fn(x: int) -> int:
            return x * 2

        with (
            patch("lexisearch.observability.middleware.get_tracer", return_value=t),
            patch("lexisearch.observability.middleware.get_metrics", return_value=m),
        ):
            result = my_fn(5)
        assert result == 10

    def test_counts_errors(self):
        m = InMemoryMetricsCollector()
        t = InMemoryTracer()

        @instrument("error_op")
        def bad_fn() -> None:
            raise RuntimeError("oops")

        with (
            patch("lexisearch.observability.middleware.get_tracer", return_value=t),
            patch("lexisearch.observability.middleware.get_metrics", return_value=m),
            pytest.raises(RuntimeError),
        ):
            bad_fn()


class TestObservabilityContext:
    def test_normal_execution(self):
        tracer = InMemoryTracer()
        metrics = InMemoryMetricsCollector()
        with ObservabilityContext(
            "test_ctx",
            latency_metric="ctx.latency",
            tracer=tracer,
            metrics=metrics,
        ) as ctx:
            ctx.set_attribute("key", "value")
            ctx.add_event("mid_event", detail="ok")
        assert tracer.span_count == 1
        spans = tracer.get_spans()
        assert spans[0].status == SpanStatus.OK
        assert len(metrics.get_metrics()) == 1  # latency histogram

    def test_error_recording(self):
        tracer = InMemoryTracer()
        metrics = InMemoryMetricsCollector()
        with (
            pytest.raises(ValueError),
            ObservabilityContext(
                "failing_ctx",
                latency_metric="ctx.latency",
                tracer=tracer,
                metrics=metrics,
            ),
        ):
            raise ValueError("test error")
        spans = tracer.get_spans()
        assert spans[0].status == SpanStatus.ERROR
        # latency + error counter
        assert len(metrics.get_metrics()) == 2


class TestGlobalRegistry:
    def test_set_and_get_tracer(self):
        original = get_tracer()
        new_tracer = InMemoryTracer()
        set_tracer(new_tracer)
        assert get_tracer() is new_tracer
        set_tracer(original)

    def test_set_and_get_metrics(self):
        original = get_metrics()
        new_metrics = InMemoryMetricsCollector()
        set_metrics(new_metrics)
        assert get_metrics() is new_metrics
        set_metrics(original)
