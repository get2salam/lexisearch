"""Base interfaces for the LexiSearch generation layer.

This module defines the abstract base classes and core data models for LLM
integration within the RAG pipeline. All concrete LLM adapters must implement
the ``BaseLLM`` interface, guaranteeing a uniform API regardless of the
underlying provider (OpenAI, Anthropic, local models, etc.).

Design principles
-----------------
* **Provider-agnostic** — callers depend only on this interface.
* **Streaming-first** — both sync and async streaming are first-class.
* **Typed** — all inputs/outputs are validated dataclasses.
* **Observable** — usage metadata is always returned for cost tracking.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator


class FinishReason(str, Enum):
    """Why a generation stopped.

    Attributes:
        STOP: The model hit a natural stop token.
        LENGTH: The maximum token limit was reached.
        CONTENT_FILTER: Content was blocked by a safety filter.
        ERROR: An error occurred during generation.
        UNKNOWN: Reason could not be determined.
    """

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


class MessageRole(str, Enum):
    """Role of a participant in a conversation.

    Attributes:
        SYSTEM: System-level instructions.
        USER: End-user turn.
        ASSISTANT: Model-generated turn.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: The role of the message sender.
        content: The text content of the message.
    """

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """Serialise to the standard ``{role, content}`` dict.

        Returns:
            Dict representation compatible with most LLM APIs.
        """
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def system(cls, content: str) -> Message:
        """Convenience constructor for a system message.

        Args:
            content: The system instruction text.

        Returns:
            A Message with SYSTEM role.
        """
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """Convenience constructor for a user message.

        Args:
            content: The user's text.

        Returns:
            A Message with USER role.
        """
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        """Convenience constructor for an assistant message.

        Args:
            content: The assistant's text.

        Returns:
            A Message with ASSISTANT role.
        """
        return cls(role=MessageRole.ASSISTANT, content=content)


@dataclass
class GenerationConfig:
    """Parameters controlling how the LLM generates text.

    Attributes:
        model: Model identifier (e.g., ``"gpt-4o"``).
        temperature: Sampling temperature in ``[0, 2]``. Lower = more
            deterministic; higher = more creative.
        max_tokens: Upper bound on tokens in the completion.
        top_p: Nucleus sampling probability mass.
        frequency_penalty: Penalise repeated tokens (OpenAI-style).
        presence_penalty: Penalise tokens already present (OpenAI-style).
        stop: One or more stop sequences.
        stream: Whether to return tokens as a stream.
        extra: Provider-specific extra parameters.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    """A request to generate text from an LLM.

    Attributes:
        messages: The conversation history sent to the model.
        config: Generation parameters.
        metadata: Arbitrary caller-supplied metadata (not sent to LLM).
    """

    messages: list[Message]
    config: GenerationConfig = field(default_factory=GenerationConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def last_user_message(self) -> str | None:
        """Return the content of the final USER message, if any.

        Returns:
            Content string or None.
        """
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER:
                return msg.content
        return None


@dataclass
class TokenUsage:
    """Token counts reported by the LLM provider.

    Attributes:
        prompt_tokens: Tokens in the prompt / context.
        completion_tokens: Tokens generated by the model.
        total_tokens: Combined count (may differ from sum due to rounding).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        """Auto-compute total when not provided."""
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class GenerationResponse:
    """A completed response from an LLM.

    Attributes:
        content: The generated text.
        model: Identifier of the model that was used.
        finish_reason: Why generation stopped.
        usage: Token usage statistics.
        latency_ms: Wall-clock latency in milliseconds.
        metadata: Provider-specific extra fields.
    """

    content: str
    model: str = ""
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """True if generation ended cleanly (STOP reason).

        Returns:
            Whether finish reason is STOP.
        """
        return self.finish_reason == FinishReason.STOP

    @property
    def is_truncated(self) -> bool:
        """True if the response was cut off by a length limit.

        Returns:
            Whether finish reason is LENGTH.
        """
        return self.finish_reason == FinishReason.LENGTH


@dataclass
class StreamChunk:
    """A single streamed token or token group from the LLM.

    Attributes:
        delta: Incremental text content for this chunk.
        finish_reason: Set only in the final chunk.
        usage: Set only in the final chunk (when reported).
        is_final: Whether this is the last chunk in the stream.
    """

    delta: str = ""
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None
    is_final: bool = False


class BaseLLM(ABC):
    """Abstract base class for all LLM adapters.

    Subclasses must implement :meth:`complete` and
    :meth:`stream`.  Async variants have default implementations
    that delegate to the sync versions — override them for true
    async performance.

    Examples:
        >>> class EchoLLM(BaseLLM):
        ...     @property
        ...     def model_name(self) -> str: return "echo"
        ...     @property
        ...     def provider(self) -> str: return "local"
        ...     def complete(self, request): ...
        ...     def stream(self, request): ...
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier.

        Returns:
            Model name string (e.g., ``"gpt-4o"``).
        """
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider name (e.g., ``"openai"``, ``"anthropic"``).

        Returns:
            Lowercase provider identifier.
        """
        ...

    @abstractmethod
    def complete(self, request: GenerationRequest) -> GenerationResponse:
        """Run a blocking chat completion.

        Args:
            request: The generation request.

        Returns:
            The completed response.

        Raises:
            LLMError: On API or network failures.
        """
        ...

    @abstractmethod
    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """Stream a chat completion token-by-token.

        Args:
            request: The generation request (``config.stream`` is
                automatically set to True).

        Yields:
            :class:`StreamChunk` objects until the stream ends.

        Raises:
            LLMError: On API or network failures.
        """
        ...

    async def acomplete(self, request: GenerationRequest) -> GenerationResponse:
        """Async chat completion (default: delegates to sync).

        Override in subclasses for true async I/O.

        Args:
            request: The generation request.

        Returns:
            The completed response.
        """
        return self.complete(request)

    async def astream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming completion (default: delegates to sync).

        Override in subclasses for true async I/O.

        Args:
            request: The generation request.

        Yields:
            :class:`StreamChunk` objects.
        """
        for chunk in self.stream(request):
            yield chunk

    def _timed_complete(
        self,
        fn: Callable[[GenerationRequest], GenerationResponse],
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Helper: call *fn* and stamp latency onto the response.

        Args:
            fn: Callable that returns a :class:`GenerationResponse`.
            request: The generation request.

        Returns:
            Response with :attr:`latency_ms` populated.
        """
        t0 = time.perf_counter()
        response = fn(request)
        response.latency_ms = (time.perf_counter() - t0) * 1000
        return response

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"{self.__class__.__name__}(model={self.model_name!r}, provider={self.provider!r})"


class LLMError(Exception):
    """Raised when an LLM API call fails.

    Attributes:
        message: Human-readable error message.
        provider: The LLM provider that raised the error.
        status_code: HTTP status code, if applicable.
        retryable: Whether retrying the request may succeed.
    """

    def __init__(
        self,
        message: str,
        provider: str = "",
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        """Initialise the error.

        Args:
            message: Human-readable error description.
            provider: LLM provider name.
            status_code: HTTP status code if available.
            retryable: Whether the caller should retry.
        """
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"LLMError({str(self)!r}, provider={self.provider!r}, "
            f"status_code={self.status_code}, retryable={self.retryable})"
        )
