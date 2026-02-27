"""Pipeline event system and hook infrastructure.

This module provides a lightweight publish-subscribe mechanism for observing
pipeline lifecycle events.  Handlers are plain callables registered with an
:class:`EventBus` instance; the bus dispatches events synchronously in
registration order.

Design principles
-----------------
* **Decoupled** — pipeline stages emit events without knowing who listens.
* **Ordered** — handlers fire in the order they were registered.
* **Safe** — per-handler errors are caught and re-raised as :class:`EventError`
  so a broken listener never silently corrupts a run.
* **Typed** — every event carries a structured :class:`PipelineEvent` payload.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Lifecycle events emitted during pipeline execution.

    Attributes:
        PIPELINE_START: The pipeline has begun processing.
        PIPELINE_FINISH: The pipeline completed successfully.
        PIPELINE_ERROR: An unrecoverable error occurred in the pipeline.
        STEP_START: A named pipeline step has started.
        STEP_FINISH: A named pipeline step completed successfully.
        STEP_ERROR: A named pipeline step raised an exception.
        PROGRESS: An incremental progress update (e.g., batch N of M).
    """

    PIPELINE_START = "pipeline_start"
    PIPELINE_FINISH = "pipeline_finish"
    PIPELINE_ERROR = "pipeline_error"
    STEP_START = "step_start"
    STEP_FINISH = "step_finish"
    STEP_ERROR = "step_error"
    PROGRESS = "progress"


@dataclass
class PipelineEvent:
    """Payload carried by every event emitted through the :class:`EventBus`.

    Attributes:
        event_type: The kind of lifecycle event.
        pipeline_id: Identifier of the pipeline instance that emitted this event.
        step: Name of the current pipeline step (empty for pipeline-level events).
        data: Arbitrary structured data relevant to the event.
        error: Exception instance, populated only for ``*_ERROR`` events.
    """

    event_type: EventType
    pipeline_id: str
    step: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: Exception | None = None

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"PipelineEvent(type={self.event_type.value!r}, "
            f"pipeline={self.pipeline_id!r}, step={self.step!r})"
        )


class EventError(Exception):
    """Raised when an event handler raises an unhandled exception.

    Attributes:
        handler_name: Qualified name of the failing handler.
        event: The event that triggered the failure.
        cause: The original exception raised by the handler.
    """

    def __init__(self, handler_name: str, event: PipelineEvent, cause: Exception) -> None:
        """Initialise an EventError.

        Args:
            handler_name: Name of the handler that raised the error.
            event: The event being dispatched when the error occurred.
            cause: The original exception.
        """
        super().__init__(
            f"Handler {handler_name!r} failed on event {event.event_type.value!r}: {cause}"
        )
        self.handler_name = handler_name
        self.event = event
        self.cause = cause


# Type alias for event handler callables.
EventHandler = Any  # Callable[[PipelineEvent], None]


@dataclass
class HandlerRegistration:
    """An entry in the event bus handler table.

    Attributes:
        handler: The callable to invoke.
        event_types: If non-empty, only these event types trigger the handler.
            An empty set means the handler receives *all* event types.
        name: Human-readable label used in logging and error messages.
    """

    handler: EventHandler
    event_types: frozenset[EventType]
    name: str


class EventBus:
    """Central dispatcher for pipeline lifecycle events.

    Handlers are callables with signature ``(event: PipelineEvent) -> None``.
    They are invoked synchronously in registration order.  Errors raised by
    handlers are wrapped in :class:`EventError` and re-raised unless
    ``raise_on_error`` is False, in which case they are logged and swallowed.

    Args:
        raise_on_error: When ``True`` (default), the first handler exception
            halts dispatch and re-raises as :class:`EventError`.  When
            ``False``, errors are logged at WARNING level and dispatch
            continues.

    Examples:
        >>> bus = EventBus()
        >>> received = []
        >>> bus.subscribe(lambda e: received.append(e.event_type))
        >>> event = PipelineEvent(EventType.PIPELINE_START, "p1")
        >>> bus.emit(event)
        >>> received[0] is EventType.PIPELINE_START
        True
    """

    def __init__(self, raise_on_error: bool = True) -> None:
        """Initialise the event bus.

        Args:
            raise_on_error: Whether to re-raise handler exceptions.
        """
        self._handlers: list[HandlerRegistration] = []
        self.raise_on_error = raise_on_error

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def subscribe(
        self,
        handler: EventHandler,
        *event_types: EventType,
        name: str = "",
    ) -> None:
        """Register a handler for one or more event types.

        Args:
            handler: Callable to invoke when a matching event is emitted.
            *event_types: Event types to filter on.  If omitted, the handler
                receives every event type.
            name: Optional human-readable label for this handler.  Defaults
                to the handler's ``__name__`` attribute when available.
        """
        label = name or getattr(handler, "__name__", repr(handler))
        self._handlers.append(
            HandlerRegistration(
                handler=handler,
                event_types=frozenset(event_types),
                name=label,
            )
        )

    def unsubscribe(self, handler: EventHandler) -> int:
        """Remove all registrations for a given handler callable.

        Args:
            handler: The callable to remove.

        Returns:
            The number of registrations removed.
        """
        before = len(self._handlers)
        self._handlers = [r for r in self._handlers if r.handler is not handler]
        return before - len(self._handlers)

    def clear(self) -> None:
        """Remove all registered handlers."""
        self._handlers.clear()

    @property
    def handler_count(self) -> int:
        """Return the number of registered handlers.

        Returns:
            Count of handler registrations.
        """
        return len(self._handlers)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def emit(self, event: PipelineEvent) -> None:
        """Dispatch *event* to all matching registered handlers.

        Args:
            event: The event to dispatch.

        Raises:
            EventError: If a handler raises and ``raise_on_error`` is True.
        """
        for reg in self._handlers:
            if reg.event_types and event.event_type not in reg.event_types:
                continue
            try:
                reg.handler(event)
            except Exception as exc:
                err = EventError(reg.name, event, exc)
                if self.raise_on_error:
                    raise err from exc
                logger.warning("Event handler %r raised: %s", reg.name, exc)

    def emit_start(self, pipeline_id: str, data: dict[str, Any] | None = None) -> None:
        """Convenience: emit a PIPELINE_START event.

        Args:
            pipeline_id: Identifier of the pipeline.
            data: Optional extra data to attach.
        """
        self.emit(
            PipelineEvent(
                event_type=EventType.PIPELINE_START,
                pipeline_id=pipeline_id,
                data=data or {},
            )
        )

    def emit_finish(self, pipeline_id: str, data: dict[str, Any] | None = None) -> None:
        """Convenience: emit a PIPELINE_FINISH event.

        Args:
            pipeline_id: Identifier of the pipeline.
            data: Optional extra data (e.g., total latency).
        """
        self.emit(
            PipelineEvent(
                event_type=EventType.PIPELINE_FINISH,
                pipeline_id=pipeline_id,
                data=data or {},
            )
        )

    def emit_error(
        self,
        pipeline_id: str,
        error: Exception,
        step: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Convenience: emit a PIPELINE_ERROR event.

        Args:
            pipeline_id: Identifier of the pipeline.
            error: The exception that caused the failure.
            step: Name of the step where the error occurred (if known).
            data: Optional additional context.
        """
        self.emit(
            PipelineEvent(
                event_type=EventType.PIPELINE_ERROR,
                pipeline_id=pipeline_id,
                step=step,
                data=data or {},
                error=error,
            )
        )

    def emit_step_start(
        self, pipeline_id: str, step: str, data: dict[str, Any] | None = None
    ) -> None:
        """Convenience: emit a STEP_START event.

        Args:
            pipeline_id: Identifier of the pipeline.
            step: Name of the step starting.
            data: Optional context data.
        """
        self.emit(
            PipelineEvent(
                event_type=EventType.STEP_START,
                pipeline_id=pipeline_id,
                step=step,
                data=data or {},
            )
        )

    def emit_step_finish(
        self, pipeline_id: str, step: str, data: dict[str, Any] | None = None
    ) -> None:
        """Convenience: emit a STEP_FINISH event.

        Args:
            pipeline_id: Identifier of the pipeline.
            step: Name of the step that finished.
            data: Optional result metadata.
        """
        self.emit(
            PipelineEvent(
                event_type=EventType.STEP_FINISH,
                pipeline_id=pipeline_id,
                step=step,
                data=data or {},
            )
        )

    def emit_progress(
        self,
        pipeline_id: str,
        step: str,
        current: int,
        total: int,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Convenience: emit a PROGRESS event.

        Args:
            pipeline_id: Identifier of the pipeline.
            step: Name of the active step.
            current: Number of items processed so far.
            total: Total number of items.
            data: Optional additional context.
        """
        self.emit(
            PipelineEvent(
                event_type=EventType.PROGRESS,
                pipeline_id=pipeline_id,
                step=step,
                data={"current": current, "total": total, **(data or {})},
            )
        )

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"EventBus(handlers={self.handler_count}, raise_on_error={self.raise_on_error})"
