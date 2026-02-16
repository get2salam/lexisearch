"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lexisearch.models import Chunk, EmbeddedChunk, Embedding


class BaseEmbedder(ABC):
    """Abstract base class for all embedding providers.

    Subclasses must implement :meth:`embed_text`, :meth:`embed_batch`,
    :meth:`model_name`, and :meth:`dimensions`.

    Attributes:
        _cache: Optional in-memory cache mapping text → vector.
    """

    def __init__(self, use_cache: bool = True) -> None:
        """Initialize the embedder.

        Args:
            use_cache: Enable in-memory embedding cache.
        """
        self.use_cache = use_cache
        self._cache: dict[str, list[float]] = {}

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: The input text.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: A list of input texts.

        Returns:
            A list of embedding vectors, one per input text.
        """
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the embedding model.

        Returns:
            Model identifier string.
        """
        ...

    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors.

        Returns:
            Number of dimensions.
        """
        ...

    def embed_text_cached(self, text: str) -> list[float]:
        """Embed text with caching support.

        Args:
            text: The input text.

        Returns:
            The embedding vector.
        """
        if self.use_cache and text in self._cache:
            return self._cache[text]
        vector = self.embed_text(text)
        if self.use_cache:
            self._cache[text] = vector
        return vector

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        """Embed a single chunk.

        Args:
            chunk: The chunk to embed.

        Returns:
            An :class:`EmbeddedChunk` containing both the chunk and its
            embedding.
        """
        vector = self.embed_text_cached(chunk.content)
        embedding = Embedding(
            chunk_id=chunk.id,
            vector=vector,
            model=self.model_name(),
            dimensions=self.dimensions(),
        )
        return EmbeddedChunk(chunk=chunk, embedding=embedding)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed multiple chunks using batch processing.

        Args:
            chunks: Chunks to embed.

        Returns:
            A list of :class:`EmbeddedChunk` objects.
        """
        # Check cache first, collect uncached
        results: list[EmbeddedChunk | None] = [None] * len(chunks)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, chunk in enumerate(chunks):
            if self.use_cache and chunk.content in self._cache:
                vector = self._cache[chunk.content]
                embedding = Embedding(
                    chunk_id=chunk.id,
                    vector=vector,
                    model=self.model_name(),
                    dimensions=self.dimensions(),
                )
                results[i] = EmbeddedChunk(chunk=chunk, embedding=embedding)
            else:
                uncached_indices.append(i)
                uncached_texts.append(chunk.content)

        # Batch embed uncached texts
        if uncached_texts:
            vectors = self.embed_batch(uncached_texts)
            for j, idx in enumerate(uncached_indices):
                chunk = chunks[idx]
                vector = vectors[j]
                if self.use_cache:
                    self._cache[chunk.content] = vector
                embedding = Embedding(
                    chunk_id=chunk.id,
                    vector=vector,
                    model=self.model_name(),
                    dimensions=self.dimensions(),
                )
                results[idx] = EmbeddedChunk(chunk=chunk, embedding=embedding)

        return [r for r in results if r is not None]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Return the number of cached embeddings.

        Returns:
            Cache entry count.
        """
        return len(self._cache)

    def get_config(self) -> dict[str, Any]:
        """Return the embedder configuration as a dictionary.

        Returns:
            Configuration dictionary.
        """
        return {
            "model": self.model_name(),
            "dimensions": self.dimensions(),
            "use_cache": self.use_cache,
            "cache_size": self.cache_size,
        }
