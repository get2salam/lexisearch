"""Streaming response handling for the LexiSearch generation layer.

Provides utilities for consuming, buffering, and processing token streams
from LLM adapters.  Supports callback hooks for real-time token delivery
(e.g., server-sent events, WebSocket pushes) and accumulation into a final
:class:`GenerationResponse`.

Components
----------
* :class:`StreamBuffer` — accumulates chunks into a mutable string buffer.
* :class:`StreamHandler` — drives a stream with optional callbacks.
* :func:`collect_stream` — one-shot collect of a complete stream.
* :func:`stream_to_response` — convert a stream into a :class:`GenerationResponse`.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from lexisearch.generation.base import (
    FinishReason,
    GenerationResponse,
    StreamChunk,
    TokenUsage,
)

# Type alias for a synchronous token callback
TokenCallback = Callable[[str], None]

# Type alias for an async token callback
AsyncTokenCallback = Callable[[str], Any]


@dataclass
class StreamBuffer:
    """Accumulates streamed token deltas into a complete string.

    Attributes:
        content: The accumulated text so far.
        chunk_count: Number of chunks received.
        finish_reason: Finish reason from the final chunk (if received).
        usage: Usage stats from the final chunk (if reported).
        is_complete: True once the final ``is_final=True`` chunk is received.

    Examples:
        >>> buf = StreamBuffer()
        >>> buf.push(StreamChunk(delta="Hello"))
        >>> buf.push(StreamChunk(delta=", world!", is_final=True, finish_reason=FinishReason.STOP))
        >>> buf.content
        'Hello, world!'
        >>> buf.is_complete
        True
    """

    content: str = ""
    chunk_count: int = 0
    finish_reason: FinishReason = FinishReason.UNKNOWN
    usage: TokenUsage | None = None
    is_complete: bool = False

    def push(self, chunk: StreamChunk) -> None:
        """Append a stream chunk to the buffer.

        Args:
            chunk: The :class:`StreamChunk` to append.
        """
        self.content += chunk.delta
        self.chunk_count += 1

        if chunk.finish_reason is not None:
            self.finish_reason = chunk.finish_reason

        if chunk.usage is not None:
            self.usage = chunk.usage

        if chunk.is_final:
            self.is_complete = True

    def reset(self) -> None:
        """Clear the buffer and reset all counters.

        Use between requests when reusing the same buffer instance.
        """
        self.content = ""
        self.chunk_count = 0
        self.finish_reason = FinishReason.UNKNOWN
        self.usage = None
        self.is_complete = False

    def to_response(self, model: str = "", latency_ms: float = 0.0) -> GenerationResponse:
        """Convert the buffered content into a :class:`GenerationResponse`.

        Args:
            model: Model identifier to attach to the response.
            latency_ms: Total latency to report.

        Returns:
            A :class:`GenerationResponse` built from accumulated content.
        """
        return GenerationResponse(
            content=self.content,
            model=model,
            finish_reason=self.finish_reason,
            usage=self.usage or TokenUsage(),
            latency_ms=latency_ms,
            metadata={"chunk_count": self.chunk_count},
        )

    def __len__(self) -> int:
        """Return the number of characters accumulated so far.

        Returns:
            Length of :attr:`content`.
        """
        return len(self.content)


class StreamHandler:
    """Drives a token stream with optional callbacks and buffering.

    Wraps an :class:`Iterator[StreamChunk]` (or async iterator) and
    provides progress callbacks, accumulation, and graceful error handling.

    Args:
        on_token: Called with each token delta string as it arrives.
        on_complete: Called with the final :class:`GenerationResponse`.
        on_error: Called with any exception that occurs mid-stream.
        model: Model name to attach to the final response.

    Examples:
        >>> tokens = []
        >>> handler = StreamHandler(on_token=tokens.append)
        >>> buf = handler.consume(iter([
        ...     StreamChunk(delta="Hi"),
        ...     StreamChunk(delta="!", is_final=True, finish_reason=FinishReason.STOP),
        ... ]))
        >>> "".join(tokens)
        'Hi!'
    """

    def __init__(
        self,
        on_token: TokenCallback | None = None,
        on_complete: Callable[[GenerationResponse], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        model: str = "",
    ) -> None:
        """Initialise the stream handler.

        Args:
            on_token: Called with each token delta as it arrives.
            on_complete: Called once with the final :class:`GenerationResponse`.
            on_error: Called with any exception that interrupts the stream.
            model: Model name to embed in the final response.
        """
        self._on_token = on_token
        self._on_complete = on_complete
        self._on_error = on_error
        self._model = model

    def consume(self, stream: Iterator[StreamChunk]) -> StreamBuffer:
        """Consume a synchronous stream, firing callbacks along the way.

        Args:
            stream: An iterator of :class:`StreamChunk` objects.

        Returns:
            The fully populated :class:`StreamBuffer` after the stream ends.
        """
        buf = StreamBuffer()
        t0 = time.perf_counter()

        try:
            for chunk in stream:
                buf.push(chunk)
                if self._on_token and chunk.delta:
                    self._on_token(chunk.delta)
        except Exception as exc:
            if self._on_error:
                self._on_error(exc)
            raise

        latency_ms = (time.perf_counter() - t0) * 1000
        if self._on_complete:
            self._on_complete(buf.to_response(model=self._model, latency_ms=latency_ms))

        return buf

    async def aconsume(self, stream: AsyncIterator[StreamChunk]) -> StreamBuffer:
        """Consume an asynchronous stream, firing callbacks along the way.

        Args:
            stream: An async iterator of :class:`StreamChunk` objects.

        Returns:
            The fully populated :class:`StreamBuffer` after the stream ends.
        """
        buf = StreamBuffer()
        t0 = time.perf_counter()

        try:
            async for chunk in stream:
                buf.push(chunk)
                if self._on_token and chunk.delta:
                    self._on_token(chunk.delta)
        except Exception as exc:
            if self._on_error:
                self._on_error(exc)
            raise

        latency_ms = (time.perf_counter() - t0) * 1000
        if self._on_complete:
            self._on_complete(buf.to_response(model=self._model, latency_ms=latency_ms))

        return buf

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"StreamHandler(model={self._model!r}, has_token_cb={self._on_token is not None})"


def collect_stream(stream: Iterator[StreamChunk]) -> str:
    """Consume a stream and return the complete response text.

    A lightweight convenience function when you only need the final text
    and don't require callbacks or metadata.

    Args:
        stream: An iterator of :class:`StreamChunk` objects.

    Returns:
        The complete concatenated response string.

    Examples:
        >>> chunks = [StreamChunk(delta="foo"), StreamChunk(delta="bar", is_final=True)]
        >>> collect_stream(iter(chunks))
        'foobar'
    """
    buf = StreamBuffer()
    for chunk in stream:
        buf.push(chunk)
    return buf.content


def stream_to_response(
    stream: Iterator[StreamChunk],
    model: str = "",
) -> GenerationResponse:
    """Consume a stream and return a complete :class:`GenerationResponse`.

    Args:
        stream: An iterator of :class:`StreamChunk` objects.
        model: Model name to embed in the response.

    Returns:
        A :class:`GenerationResponse` built from the accumulated stream.

    Examples:
        >>> chunks = [
        ...     StreamChunk(delta="42", is_final=True, finish_reason=FinishReason.STOP)
        ... ]
        >>> resp = stream_to_response(iter(chunks), model="mock")
        >>> resp.content
        '42'
        >>> resp.is_complete
        True
    """
    buf = StreamBuffer()
    t0 = time.perf_counter()
    for chunk in stream:
        buf.push(chunk)
    latency_ms = (time.perf_counter() - t0) * 1000
    return buf.to_response(model=model, latency_ms=latency_ms)


@dataclass
class ThrottledStream:
    """Wraps a stream and re-yields chunks with a minimum interval.

    Useful for rate-limiting token delivery to downstream clients such as
    WebSocket connections or SSE endpoints.

    Args:
        source: The underlying stream to throttle.
        min_interval_seconds: Minimum seconds between chunk yields.

    Examples:
        >>> import time
        >>> chunks = [StreamChunk(delta="a"), StreamChunk(delta="b", is_final=True)]
        >>> ts = ThrottledStream(iter(chunks), min_interval_seconds=0.0)
        >>> list(ts)  # doctest: +ELLIPSIS
        [StreamChunk(delta='a', ...), StreamChunk(delta='b', ...)]
    """

    source: Iterator[StreamChunk]
    min_interval_seconds: float = 0.01
    _last_yield: float = field(default_factory=time.perf_counter, init=False)

    def __iter__(self) -> Iterator[StreamChunk]:
        """Yield chunks, sleeping when necessary to enforce the minimum interval.

        Yields:
            :class:`StreamChunk` objects from the wrapped source.
        """
        for chunk in self.source:
            now = time.perf_counter()
            elapsed = now - self._last_yield
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)
            yield chunk
            self._last_yield = time.perf_counter()
