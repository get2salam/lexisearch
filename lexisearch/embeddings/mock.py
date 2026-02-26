"""Mock embedding provider for testing and development."""

from __future__ import annotations

import hashlib
import struct

from lexisearch.embeddings.base import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """Deterministic mock embedder for testing.

    Generates reproducible pseudo-random vectors from the input text hash.
    Useful for unit tests and development without requiring an API key or
    GPU.

    Args:
        dims: Dimensionality of output vectors.
        use_cache: Enable in-memory caching.

    Example:
        >>> embedder = MockEmbedder(dimensions=128)
        >>> vec = embedder.embed_text("hello")
        >>> len(vec)
        128
        >>> embedder.embed_text("hello") == vec  # deterministic
        True
    """

    def __init__(self, dimensions: int = 384, use_cache: bool = True) -> None:
        """Initialize the mock embedder.

        Args:
            dimensions: Output vector dimensionality.
            use_cache: Enable embedding cache.
        """
        super().__init__(use_cache=use_cache)
        self._dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        """Generate a deterministic pseudo-random embedding.

        The vector is derived from a SHA-256 hash of the input text,
        ensuring reproducible results.

        Args:
            text: Input text.

        Returns:
            A list of floats of length ``dimensions``.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Use the hash to seed a simple PRNG
        values: list[float] = []
        for i in range(self._dimensions):
            # Cycle through hash bytes and apply simple transformation
            i % len(digest)
            # Combine with index for variety
            seed_bytes = hashlib.sha256(
                digest + struct.pack(">I", i)
            ).digest()[:4]
            raw = struct.unpack(">I", seed_bytes)[0]
            # Normalize to [-1, 1]
            value = (raw / (2**32 - 1)) * 2 - 1
            values.append(value)

        # L2-normalize
        norm = sum(v * v for v in values) ** 0.5
        if norm > 0:
            values = [v / norm for v in values]

        return values

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: Input texts.

        Returns:
            A list of embedding vectors.
        """
        return [self.embed_text(text) for text in texts]

    def model_name(self) -> str:
        """Return the mock model name.

        Returns:
            ``"mock-embedder"``
        """
        return "mock-embedder"

    def dimensions(self) -> int:
        """Return the configured dimensionality.

        Returns:
            Number of dimensions.
        """
        return self._dimensions
