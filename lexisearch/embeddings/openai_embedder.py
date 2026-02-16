"""OpenAI embedding provider.

Requires the ``openai`` extra: ``pip install lexisearch[openai]``.
"""

from __future__ import annotations

from typing import Any

from lexisearch.embeddings.base import BaseEmbedder

_OPENAI_AVAILABLE: bool
try:
    import openai  # type: ignore[import-untyped]

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class OpenAIEmbedder(BaseEmbedder):
    """Embedding provider backed by the OpenAI Embeddings API.

    Uses the ``openai`` Python client to generate embeddings.  Requires
    an API key set via the ``OPENAI_API_KEY`` environment variable or
    passed explicitly.

    Args:
        model: OpenAI model name (default ``"text-embedding-3-small"``).
        api_key: Explicit API key. If ``None``, the client reads from
            the environment.
        dims: Expected embedding dimensionality.
        use_cache: Enable in-memory caching.

    Raises:
        ImportError: If the ``openai`` package is not installed.

    Example:
        >>> embedder = OpenAIEmbedder(model="text-embedding-3-small")
        >>> vec = embedder.embed_text("Hello, world!")
    """

    _DEFAULT_MODEL = "text-embedding-3-small"
    _DEFAULT_DIMS = 1536

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        dims: int = _DEFAULT_DIMS,
        use_cache: bool = True,
    ) -> None:
        """Initialize the OpenAI embedder.

        Args:
            model: OpenAI model identifier.
            api_key: Optional explicit API key.
            dims: Expected output dimensionality.
            use_cache: Enable embedding cache.

        Raises:
            ImportError: If ``openai`` is not installed.
        """
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "The openai package is required for OpenAIEmbedder. "
                "Install it with: pip install lexisearch[openai]"
            )
        super().__init__(use_cache=use_cache)
        self._model = model
        self._dims = dims

        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client: Any = openai.OpenAI(**client_kwargs)

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding using the OpenAI API.

        Args:
            text: Input text.

        Returns:
            Embedding vector as a list of floats.
        """
        response = self._client.embeddings.create(
            input=[text],
            model=self._model,
        )
        return list(response.data[0].embedding)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        The OpenAI API supports batched requests natively.

        Args:
            texts: Input texts.

        Returns:
            A list of embedding vectors.
        """
        if not texts:
            return []

        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [list(d.embedding) for d in sorted_data]

    def model_name(self) -> str:
        """Return the OpenAI model name.

        Returns:
            The configured model identifier.
        """
        return self._model

    def dimensions(self) -> int:
        """Return the expected dimensionality.

        Returns:
            Number of dimensions.
        """
        return self._dims
