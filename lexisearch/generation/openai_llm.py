"""OpenAI LLM adapter for the LexiSearch generation layer.

Provides a production-ready :class:`OpenAILLM` adapter that wraps the
OpenAI Chat Completions API.  Requires the ``openai`` package (>=1.0).

The adapter is intentionally thin: it delegates all retry logic, timeouts,
and connection pooling to the ``openai`` SDK.  You can inject a custom
``openai.OpenAI`` client for testing or to share a client across adapters.

Features
--------
* Sync and async completions via ``openai`` v1 client.
* Streaming support with ``StreamChunk`` yielding.
* Automatic usage extraction from API responses.
* Configurable default model and generation config.

Usage::

    from lexisearch.generation.openai_llm import OpenAILLM
    from lexisearch.generation.base import GenerationRequest, Message

    llm = OpenAILLM(api_key="sk-...")
    request = GenerationRequest(messages=[Message.user("Explain RAG in one sentence.")])
    response = llm.complete(request)
    print(response.content)
"""

from __future__ import annotations

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

    pass


def _finish_reason(raw: str | None) -> FinishReason:
    """Map an OpenAI finish-reason string to :class:`FinishReason`.

    Args:
        raw: The raw ``finish_reason`` string from the API.

    Returns:
        The corresponding :class:`FinishReason` enum value.
    """
    _map = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "content_filter": FinishReason.CONTENT_FILTER,
    }
    return _map.get(raw or "", FinishReason.UNKNOWN)


class OpenAILLM(BaseLLM):
    """LLM adapter for the OpenAI Chat Completions API.

    Args:
        api_key: OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable when not supplied.
        default_model: Model to use when the request config does not
            specify one (default ``"gpt-4o-mini"``).
        default_config: Base :class:`GenerationConfig` applied to every
            request before request-level overrides.
        client: Optional pre-constructed ``openai.OpenAI`` instance.
            Useful for sharing a single client across adapters or injecting
            a mock in tests.
        base_url: Optional base URL override (e.g., for Azure OpenAI or
            local proxies).

    Raises:
        ImportError: If the ``openai`` package is not installed.
        LLMError: On API authentication, rate-limit, or server errors.

    Examples:
        >>> # With a real key:
        >>> # llm = OpenAILLM(api_key="sk-...")
        >>> # response = llm.complete(GenerationRequest(messages=[Message.user("Hi")]))

        >>> # In tests, inject a mock client to avoid real API calls.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gpt-4o-mini",
        default_config: GenerationConfig | None = None,
        client: Any | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialise the OpenAI LLM adapter.

        Args:
            api_key: OpenAI API key.
            default_model: Fallback model when request config omits one.
            default_config: Base generation config applied to every request.
            client: Pre-constructed ``openai.OpenAI`` client (for testing).
            base_url: Optional base URL override.
        """
        self._default_model = default_model
        self._default_config = default_config or GenerationConfig(model=default_model)
        self._client = client  # Injected client (may be None until first use)
        self._api_key = api_key
        self._base_url = base_url

    @property
    def model_name(self) -> str:
        """Canonical model identifier.

        Returns:
            The default model name.
        """
        return self._default_model

    @property
    def provider(self) -> str:
        """Provider identifier.

        Returns:
            Always ``"openai"``.
        """
        return "openai"

    def _get_client(self) -> Any:
        """Return (or lazily construct) the OpenAI client.

        Returns:
            An ``openai.OpenAI`` client instance.

        Raises:
            ImportError: If ``openai`` is not installed.
        """
        if self._client is not None:
            return self._client

        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAILLM. "
                "Install it with: pip install openai"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url

        self._client = openai.OpenAI(**kwargs)
        return self._client

    def _build_params(self, request: GenerationRequest) -> dict[str, Any]:
        """Build the keyword arguments for the OpenAI API call.

        Args:
            request: The generation request.

        Returns:
            Dict suitable for ``client.chat.completions.create(**params)``.
        """
        cfg = request.config
        params: dict[str, Any] = {
            "model": cfg.model or self._default_model,
            "messages": [m.to_dict() for m in request.messages],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
        }
        if cfg.frequency_penalty:
            params["frequency_penalty"] = cfg.frequency_penalty
        if cfg.presence_penalty:
            params["presence_penalty"] = cfg.presence_penalty
        if cfg.stop:
            params["stop"] = cfg.stop
        params.update(cfg.extra)
        return params

    def complete(self, request: GenerationRequest) -> GenerationResponse:
        """Run a blocking chat completion via the OpenAI API.

        Args:
            request: The generation request.

        Returns:
            A :class:`GenerationResponse` with the model's reply.

        Raises:
            LLMError: On API errors (auth, rate limit, server fault).
        """
        import time

        client = self._get_client()
        params = self._build_params(request)
        params.pop("stream", None)  # Ensure non-streaming

        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(**params)
            latency_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        choice = resp.choices[0]
        usage = resp.usage

        return GenerationResponse(
            content=choice.message.content or "",
            model=resp.model,
            finish_reason=_finish_reason(choice.finish_reason),
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            latency_ms=latency_ms,
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """Stream a chat completion token-by-token via the OpenAI API.

        Args:
            request: The generation request.

        Yields:
            :class:`StreamChunk` objects as tokens arrive.

        Raises:
            LLMError: On API errors.
        """
        client = self._get_client()
        params = self._build_params(request)
        params["stream"] = True

        try:
            with client.chat.completions.create(**params) as stream_resp:
                for event in stream_resp:
                    if not event.choices:
                        continue
                    choice = event.choices[0]
                    delta = choice.delta.content or ""
                    raw_finish = choice.finish_reason
                    finish = _finish_reason(raw_finish) if raw_finish else None
                    is_final = raw_finish is not None
                    yield StreamChunk(
                        delta=delta,
                        finish_reason=finish,
                        is_final=is_final,
                    )
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    @staticmethod
    def _wrap_error(exc: Exception) -> LLMError:
        """Wrap a provider SDK exception in a :class:`LLMError`.

        Args:
            exc: The original exception from the OpenAI SDK.

        Returns:
            A :class:`LLMError` with provider and retryability metadata.
        """
        msg = str(exc)
        exc_type = type(exc).__name__

        retryable = any(
            hint in exc_type.lower() for hint in ("ratelimit", "timeout", "apierror", "connection")
        )
        status_code: int | None = getattr(exc, "status_code", None)

        return LLMError(
            message=f"OpenAI API error ({exc_type}): {msg}",
            provider="openai",
            status_code=status_code,
            retryable=retryable,
        )
