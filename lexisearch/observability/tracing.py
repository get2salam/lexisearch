"""In-process distributed tracing implementation."""

from __future__ import annotations

import uuid
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from lexisearch.observability.base import BaseTracer, Span, SpanStatus

if TYPE_CHECKING:
    from collections.abc import Generator

# ---------------------------------------------------------------------------
# No-op tracer (zero overhead default)
# ---------------------------------------------------------------------------


class NoOpTracer(BaseTracer):
    """A tracer that does nothing — safe as a default when tracing is disabled."""

    def start_span(
        self,
        name: str,
        *,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Create a no-op span."""
        return Span(
            name=name,
            trace_id="noop",
            span_id="noop",
            parent_id=parent_id,
            attributes=attributes or {},
        )

    def finish_span(self, span: Span, *, status: SpanStatus = SpanStatus.OK) -> None:
        """No-op finish — discard the span."""
        span.finish(status=status)

    def get_spans(self, trace_id: str | None = None) -> list[Span]:
        """Return empty list — no spans are collected."""
        return []


# ---------------------------------------------------------------------------
# In-memory tracer
# ---------------------------------------------------------------------------


class InMemoryTracer(BaseTracer):
    """Collects spans in memory.

    Suitable for testing, development, and short-lived processes.  Not
    thread-safe for high-concurrency production usage — use an OTLP exporter
    for that.

    Attributes:
        max_spans: Maximum number of spans to retain (FIFO eviction).
    """

    def __init__(self, max_spans: int = 10_000) -> None:
        """Initialise the in-memory tracer.

        Args:
            max_spans: Maximum spans to keep before evicting oldest.
        """
        self.max_spans = max_spans
        self._spans: list[Span] = []
        self._by_trace: dict[str, list[Span]] = defaultdict(list)

    def start_span(
        self,
        name: str,
        *,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span and return it (not yet stored until finished).

        Args:
            name: Human-readable operation name.
            parent_id: Parent span ID for nested operations.
            attributes: Initial key-value attributes.

        Returns:
            A new, open :class:`Span`.
        """
        trace_id = parent_id.split(".")[0] if parent_id else uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
        return Span(
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            attributes=attributes or {},
        )

    def finish_span(self, span: Span, *, status: SpanStatus = SpanStatus.OK) -> None:
        """Finish a span and store it in memory.

        Args:
            span: The span to finish.
            status: Final completion status.
        """
        span.finish(status=status)
        # Evict oldest if at capacity
        if len(self._spans) >= self.max_spans:
            oldest = self._spans.pop(0)
            trace_spans = self._by_trace.get(oldest.trace_id, [])
            if oldest in trace_spans:
                trace_spans.remove(oldest)
        self._spans.append(span)
        self._by_trace[span.trace_id].append(span)

    def get_spans(self, trace_id: str | None = None) -> list[Span]:
        """Return collected spans.

        Args:
            trace_id: If provided, return only spans for this trace.

        Returns:
            List of :class:`Span` objects (newest last).
        """
        if trace_id is not None:
            return list(self._by_trace.get(trace_id, []))
        return list(self._spans)

    def clear(self) -> None:
        """Remove all stored spans."""
        self._spans.clear()
        self._by_trace.clear()

    @property
    def span_count(self) -> int:
        """Total number of stored spans."""
        return len(self._spans)

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager that starts and finishes a span automatically.

        Args:
            name: Operation name.
            parent_id: Optional parent span ID.
            attributes: Initial attributes.

        Yields:
            The open :class:`Span` — add attributes/events inside the block.

        Example:
            >>> tracer = InMemoryTracer()
            >>> with tracer.trace("embed") as span:
            ...     span.set_attribute("model", "text-embedding-ada-002")
        """
        span = self.start_span(name, parent_id=parent_id, attributes=attributes)
        try:
            yield span
        except Exception as exc:
            span.add_event("error", message=str(exc), exc_type=type(exc).__name__)
            self.finish_span(span, status=SpanStatus.ERROR)
            raise
        else:
            self.finish_span(span, status=SpanStatus.OK)
