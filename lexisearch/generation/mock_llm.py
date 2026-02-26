"""Mock LLM for deterministic testing and offline development.

Provides a fully functional :class:`BaseLLM` implementation that returns
configured or auto-generated responses without any network calls.  Use this
in unit tests, CI pipelines, and demos where real API keys are unavailable.

Features
--------
* Configurable static responses.
* Controllable latency simulation.
* Streaming simulation with token-by-token yields.
* Request capture for assertion in tests.
* Error injection for failure-path testing.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from lexisearch.generation.base import (
    BaseLLM,
    FinishReason,
    GenerationConfig,
    GenerationRequest,
    GenerationResponse,
    LLMError,
    StreamChunk,
    TokenUsage,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class MockLLM(BaseLLM):
    """A deterministic mock LLM for testing.

    Args:
        response_text: Fixed text to return for every request.
            Defaults to a templated response using the last user message.
        model: Model identifier to advertise.
        latency_seconds: Simulated network latency in seconds.
        raise_error: If set, raise this error on every call.
        stream_chunk_size: Number of characters per stream chunk.

    Examples:
        >>> llm = MockLLM(response_text="The answer is 42.")
        >>> req = GenerationRequest(messages=[Message.user("What is the answer?")])
        >>> resp = llm.complete(req)
        >>> resp.content
        'The answer is 42.'
        >>> resp.finish_reason
        <FinishReason.STOP: 'stop'>
    """

    def __init__(
        self,
        response_text: str | None = None,
        model: str = "mock-llm-v1",
        latency_seconds: float = 0.0,
        raise_error: LLMError | None = None,
        stream_chunk_size: int = 5,
    ) -> None:
        """Initialise the mock LLM with configurable behaviour.

        Args:
            response_text: Fixed response text.  When ``None`` a templated
                message is generated from the last user message.
            model: Advertised model identifier.
            latency_seconds: Simulated latency in seconds.
            raise_error: If not ``None``, raise this on every call.
            stream_chunk_size: Characters per streamed chunk.
        """
        self._response_text = response_text
        self._model = model
        self._latency_seconds = latency_seconds
        self._raise_error = raise_error
        self._stream_chunk_size = stream_chunk_size

        # Captured requests for test assertions
        self.captured_requests: list[GenerationRequest] = []
        self.call_count: int = 0

    @property
    def model_name(self) -> str:
        """Canonical model identifier.

        Returns:
            The mock model name.
        """
        return self._model

    @property
    def provider(self) -> str:
        """Provider identifier.

        Returns:
            Always ``"mock"``.
        """
        return "mock"

    def _build_response_text(self, request: GenerationRequest) -> str:
        """Build the response text for a given request.

        If a fixed ``response_text`` was provided at construction time,
        it is returned verbatim.  Otherwise a templated response is
        generated from the last user message.

        Args:
            request: The incoming generation request.

        Returns:
            The response content string.
        """
        if self._response_text is not None:
            return self._response_text
        query = request.last_user_message or "your question"
        return (
            f"This is a mock response to: {query!r}. "
            "No real LLM was called; this text is generated deterministically "
            "by MockLLM for testing purposes."
        )

    def complete(self, request: GenerationRequest) -> GenerationResponse:
        """Return a deterministic completion.

        Args:
            request: The generation request.

        Returns:
            A :class:`GenerationResponse` with the configured text.

        Raises:
            LLMError: If ``raise_error`` was set at construction.
        """
        self.captured_requests.append(request)
        self.call_count += 1

        if self._raise_error is not None:
            raise self._raise_error

        if self._latency_seconds > 0:
            time.sleep(self._latency_seconds)

        content = self._build_response_text(request)
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        completion_tokens = len(content.split())

        return GenerationResponse(
            content=content,
            model=self._model,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            metadata={"mock": True},
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """Yield the response text in small chunks.

        Args:
            request: The generation request.

        Yields:
            :class:`StreamChunk` objects, one per chunk of text, then a
            final chunk with ``is_final=True``.

        Raises:
            LLMError: If ``raise_error`` was set at construction.
        """
        self.captured_requests.append(request)
        self.call_count += 1

        if self._raise_error is not None:
            raise self._raise_error

        content = self._build_response_text(request)
        chunk_size = max(1, self._stream_chunk_size)

        for i in range(0, len(content), chunk_size):
            delta = content[i : i + chunk_size]
            is_last_text = i + chunk_size >= len(content)
            if self._latency_seconds > 0:
                time.sleep(self._latency_seconds / max(1, len(content) // chunk_size))
            yield StreamChunk(
                delta=delta,
                finish_reason=FinishReason.STOP if is_last_text else None,
                is_final=is_last_text,
            )

        # Ensure we always emit a final chunk even for empty content
        if not content:
            yield StreamChunk(
                delta="",
                finish_reason=FinishReason.STOP,
                is_final=True,
            )

    def reset(self) -> None:
        """Clear captured requests and reset call counter.

        Use this between test cases to avoid state bleed.

        Examples:
            >>> llm = MockLLM()
            >>> _ = llm.complete(GenerationRequest(messages=[Message.user("Hi")]))
            >>> llm.call_count
            1
            >>> llm.reset()
            >>> llm.call_count
            0
        """
        self.captured_requests.clear()
        self.call_count = 0

    def set_response(self, text: str) -> None:
        """Update the fixed response text at runtime.

        Args:
            text: New response text to use for subsequent calls.
        """
        self._response_text = text

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"MockLLM(model={self._model!r}, calls={self.call_count}, "
            f"has_error={self._raise_error is not None})"
        )


def make_config(**kwargs: Any) -> GenerationConfig:
    """Convenience factory for :class:`GenerationConfig` in tests.

    Args:
        **kwargs: Any :class:`GenerationConfig` field values.

    Returns:
        A :class:`GenerationConfig` with the given overrides.

    Examples:
        >>> cfg = make_config(temperature=0.7, max_tokens=512)
        >>> cfg.temperature
        0.7
    """
    return GenerationConfig(**kwargs)
