"""Tests for lexisearch.embeddings providers."""

from __future__ import annotations

import pytest

from lexisearch.embeddings.mock import MockEmbedder
from lexisearch.models import Chunk, ChunkStrategy


class TestMockEmbedder:
    """Tests for MockEmbedder."""

    def test_embed_text_dimensions(self) -> None:
        """Output vector should match configured dimensions."""
        embedder = MockEmbedder(dimensions=128)
        vec = embedder.embed_text("hello")
        assert len(vec) == 128

    def test_embed_text_deterministic(self) -> None:
        """Same input should produce the same vector."""
        embedder = MockEmbedder(dimensions=64, use_cache=False)
        v1 = embedder.embed_text("test")
        v2 = embedder.embed_text("test")
        assert v1 == v2

    def test_embed_text_different_inputs(self) -> None:
        """Different inputs should produce different vectors."""
        embedder = MockEmbedder(dimensions=64, use_cache=False)
        v1 = embedder.embed_text("alpha")
        v2 = embedder.embed_text("beta")
        assert v1 != v2

    def test_embed_text_normalized(self) -> None:
        """Output vectors should be approximately L2-normalized."""
        embedder = MockEmbedder(dimensions=256, use_cache=False)
        vec = embedder.embed_text("normalize me")
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_embed_batch(self) -> None:
        """embed_batch should return one vector per input."""
        embedder = MockEmbedder(dimensions=64)
        texts = ["one", "two", "three"]
        results = embedder.embed_batch(texts)
        assert len(results) == 3
        for vec in results:
            assert len(vec) == 64

    def test_embed_chunk(self, sample_chunk: Chunk) -> None:
        """embed_chunk should return an EmbeddedChunk."""
        embedder = MockEmbedder(dimensions=64)
        ec = embedder.embed_chunk(sample_chunk)
        assert ec.chunk.id == sample_chunk.id
        assert ec.embedding.dimensions == 64
        assert ec.embedding.model == "mock-embedder"

    def test_embed_chunks(self) -> None:
        """embed_chunks should batch-process multiple chunks."""
        chunks = [
            Chunk(content=f"chunk {i}", document_id="d1", index=i)
            for i in range(5)
        ]
        embedder = MockEmbedder(dimensions=32)
        results = embedder.embed_chunks(chunks)
        assert len(results) == 5

    def test_cache_hit(self) -> None:
        """Cached embeddings should be reused."""
        embedder = MockEmbedder(dimensions=32, use_cache=True)
        embedder.embed_text("cached")
        assert embedder.cache_size == 1
        # Second call should use cache
        v = embedder.embed_text_cached("cached")
        assert len(v) == 32
        assert embedder.cache_size == 1

    def test_cache_disabled(self) -> None:
        """With cache disabled, cache_size should stay 0."""
        embedder = MockEmbedder(dimensions=32, use_cache=False)
        embedder.embed_text_cached("no cache")
        assert embedder.cache_size == 0

    def test_clear_cache(self) -> None:
        """clear_cache should empty the cache."""
        embedder = MockEmbedder(dimensions=32, use_cache=True)
        embedder.embed_text_cached("a")
        embedder.embed_text_cached("b")
        assert embedder.cache_size == 2
        embedder.clear_cache()
        assert embedder.cache_size == 0

    def test_model_name(self) -> None:
        """model_name should return 'mock-embedder'."""
        embedder = MockEmbedder()
        assert embedder.model_name() == "mock-embedder"

    def test_get_config(self) -> None:
        """get_config should return a valid configuration dict."""
        embedder = MockEmbedder(dimensions=128, use_cache=True)
        config = embedder.get_config()
        assert config["model"] == "mock-embedder"
        assert config["dimensions"] == 128
        assert config["use_cache"] is True

    def test_embed_chunks_uses_cache(self) -> None:
        """embed_chunks should use cache for repeated content."""
        chunks = [
            Chunk(content="same text", document_id="d1", index=0),
            Chunk(content="same text", document_id="d1", index=1),
        ]
        embedder = MockEmbedder(dimensions=16, use_cache=True)
        results = embedder.embed_chunks(chunks)
        assert len(results) == 2
        # Both should have the same vector (same content)
        assert results[0].embedding.vector == results[1].embedding.vector
