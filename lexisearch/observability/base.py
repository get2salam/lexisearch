"""Base interfaces for the observability layer."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SpanStatus(str, Enum):
    """Completion status of a tracing span."""

    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"


class MetricKind(str, Enum):
    """Supported metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


# ---------------------------------------------------------------------------
# Span / Trace primitives
# ---------------------------------------------------------------------------


@dataclass
class SpanEvent:
    """A timestamped event attached to a span."""

    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A single unit of traced work.

    Spans are created by :class:`BaseTracer` and form the building blocks of
    distributed traces.  Each span tracks start/end times, status, and
    arbitrary key-value attributes.
    """

    name: str
    trace_id: str
    span_id: str
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Elapsed time in milliseconds, or ``None`` if still open."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1_000

    def set_attribute(self, key: str, value: Any) -> None:
        """Add or overwrite a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, **attrs: Any) -> None:
        """Attach a named event to this span."""
        self.events.append(SpanEvent(name=name, attributes=dict(attrs)))

    def finish(self, *, status: SpanStatus = SpanStatus.OK) -> None:
        """Mark the span as finished."""
        self.end_time = time.time()
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        """Serialise span to a plain dictionary."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
        }


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


@dataclass
class MetricPoint:
    """A single recorded metric observation."""

    name: str
    kind: MetricKind
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise metric point to a plain dictionary."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
        }


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class BaseTracer(ABC):
    """Abstract tracer interface.

    Implement this to create custom span exporters (OTLP, Jaeger, logging, …).
    """

    @abstractmethod
    def start_span(
        self,
        name: str,
        *,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Create and start a new span.

        Parameters
        ----------
        name:
            Human-readable operation name.
        parent_id:
            Optional parent span ID for nested operations.

        Attributes:
            Initial key-value attributes to attach.
        """

    @abstractmethod
    def finish_span(self, span: Span, *, status: SpanStatus = SpanStatus.OK) -> None:
        """Finish and export a span.

        Parameters
        ----------
        span:
            The span to finish.
        status:
            Final status (OK, ERROR, or CANCELLED).
        """

    @abstractmethod
    def get_spans(self, trace_id: str | None = None) -> list[Span]:
        """Return collected spans, optionally filtered by trace ID."""


class BaseMetricsCollector(ABC):
    """Abstract metrics collector interface."""

    @abstractmethod
    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Increment a counter metric.

        Parameters
        ----------
        name:
            Metric name (e.g. ``"rag.queries.total"``).
        value:
            Amount to add (default 1).
        **labels:
            Arbitrary label key-value pairs.
        """

    @abstractmethod
    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Record an absolute gauge value.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Current value of the gauge.
        **labels:
            Arbitrary label key-value pairs.
        """

    @abstractmethod
    def histogram(self, name: str, value: float, **labels: str) -> None:
        """Record a value in a histogram (e.g. latency, token count).

        Parameters
        ----------
        name:
            Metric name.
        value:
            Observed value.
        **labels:
            Arbitrary label key-value pairs.
        """

    @abstractmethod
    def get_metrics(self) -> list[MetricPoint]:
        """Return all recorded metric points."""
