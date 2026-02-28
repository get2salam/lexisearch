"""Sentence Transformers embedding provider.

Requires the ``sentence-transformers`` extra:
``pip install lexisearch[sentence-transformers]``.
"""

from __future__ import annotations

from typing import Any

from lexisearch.embeddings.base import BaseEmbedder

_SBERT_AVAILABLE: bool
try:
    from sentence_transformers import SentenceTransformer

    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedding provider backed by Sentence Transformers.

    Loads a local or Hugging Face-hosted model and runs inference on CPU
    or GPU.  This is ideal for offline or on-premises deployments.

    Args:
        model_name_or_path: Model identifier or local path.
            Default: ``"all-MiniLM-L6-v2"``.
        device: Torch device string (``"cpu"``, ``"cuda"``, etc.).
        batch_size: Batch size for :meth:`embed_batch`.
        use_cache: Enable in-memory caching.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.

    Example:
        >>> embedder = SentenceTransformerEmbedder()
        >>> vec = embedder.embed_text("Hello!")
    """

    _DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 64,
        use_cache: bool = True,
    ) -> None:
        """Initialize the Sentence Transformer embedder.

        Args:
            model_name_or_path: Model name or path.
            device: Torch device. ``None`` = auto-detect.
            batch_size: Batch encoding size.
            use_cache: Enable embedding cache.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        if not _SBERT_AVAILABLE:
            raise ImportError(
                "The sentence-transformers package is required. "
                "Install it with: pip install lexisearch[sentence-transformers]"
            )
        super().__init__(use_cache=use_cache)
        self._model_id = model_name_or_path
        self._batch_size = batch_size

        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        self._model: Any = SentenceTransformer(model_name_or_path, **kwargs)
        dims = self._model.get_sentence_embedding_dimension()
        self._dims: int = dims if dims is not None else 0

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding using the Sentence Transformer model.

        Args:
            text: Input text.

        Returns:
            Embedding vector as a list of floats.
        """
        vector = self._model.encode(text, show_progress_bar=False)
        return vector.tolist()  # type: ignore[no-any-return]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: Input texts.

        Returns:
            A list of embedding vectors.
        """
        if not texts:
            return []

        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def model_name(self) -> str:
        """Return the Sentence Transformer model name.

        Returns:
            The model identifier.
        """
        return self._model_id

    def dimensions(self) -> int:
        """Return the model's embedding dimensionality.

        Returns:
            Number of dimensions.
        """
        return self._dims
