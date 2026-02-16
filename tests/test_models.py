"""Tests for lexisearch.models."""

from __future__ import annotations

import pytest

from lexisearch.models import (
    Chunk,
    ChunkStrategy,
    Document,
    DocumentFormat,
    DocumentMetadata,
    EmbeddedChunk,
    Embedding,
    SearchResponse,
    SearchResult,
)


class TestDocumentMetadata:
    """Tests for DocumentMetadata."""

    def test_defaults(self) -> None:
        """Metadata should have sensible defaults."""
        meta = DocumentMetadata()
        assert meta.source == ""
        assert meta.language == "en"
        assert meta.format == DocumentFormat.UNKNOWN
        assert meta.extra == {}

    def test_custom_fields(self) -> None:
        """Metadata should accept custom fields."""
        meta = DocumentMetadata(
            source="/docs/readme.txt",
            title="Readme",
            author="Author",
            language="de",
            format=DocumentFormat.TEXT,
            extra={"version": 2},
        )
        assert meta.source == "/docs/readme.txt"
        assert meta.extra["version"] == 2


class TestDocument:
    """Tests for Document."""

    def test_creation(self, sample_document: Document) -> None:
        """Document should be created with content and metadata."""
        assert len(sample_document.content) > 0
        assert sample_document.metadata.title == "Test Document"

    def test_unique_ids(self) -> None:
        """Each document should get a unique ID."""
        d1 = Document(content="a")
        d2 = Document(content="a")
        assert d1.id != d2.id

    def test_content_hash_deterministic(self) -> None:
        """Same content should produce the same hash."""
        d1 = Document(content="test content")
        d2 = Document(content="test content")
        assert d1.content_hash == d2.content_hash

    def test_content_hash_changes(self) -> None:
        """Different content should produce different hashes."""
        d1 = Document(content="alpha")
        d2 = Document(content="beta")
        assert d1.content_hash != d2.content_hash

    def test_char_count(self) -> None:
        """char_count should return the content length."""
        doc = Document(content="hello")
        assert doc.char_count == 5

    def test_word_count(self) -> None:
        """word_count should count whitespace-separated words."""
        doc = Document(content="one two three")
        assert doc.word_count == 3

    def test_repr(self) -> None:
        """repr should be concise and include the ID."""
        doc = Document(content="short")
        r = repr(doc)
        assert "Document" in r
        assert doc.id in r


class TestChunk:
    """Tests for Chunk."""

    def test_creation(self, sample_chunk: Chunk) -> None:
        """Chunk should store content and document reference."""
        assert "Machine learning" in sample_chunk.content
        assert sample_chunk.index == 0

    def test_content_hash(self, sample_chunk: Chunk) -> None:
        """Chunk should have a deterministic content hash."""
        h = sample_chunk.content_hash
        assert len(h) == 64  # SHA-256 hex

    def test_char_count(self, sample_chunk: Chunk) -> None:
        """char_count should return content length."""
        assert sample_chunk.char_count == len(sample_chunk.content)

    def test_token_estimate(self) -> None:
        """token_estimate should return at least 1."""
        chunk = Chunk(content="hi", document_id="d1")
        assert chunk.token_estimate >= 1

    def test_repr(self, sample_chunk: Chunk) -> None:
        """repr should be concise."""
        r = repr(sample_chunk)
        assert "Chunk" in r


class TestEmbedding:
    """Tests for Embedding."""

    def test_auto_dimensions(self) -> None:
        """Dimensions should be inferred from vector length."""
        emb = Embedding(chunk_id="c1", vector=[0.1, 0.2, 0.3])
        assert emb.dimensions == 3

    def test_explicit_dimensions(self) -> None:
        """Explicit dimensions should override auto-detection."""
        emb = Embedding(chunk_id="c1", vector=[0.1, 0.2], dimensions=2)
        assert emb.dimensions == 2

    def test_norm(self) -> None:
        """norm should compute L2 norm correctly."""
        emb = Embedding(chunk_id="c1", vector=[3.0, 4.0])
        assert abs(emb.norm - 5.0) < 1e-6

    def test_repr(self) -> None:
        """repr should include dimensions and model."""
        emb = Embedding(chunk_id="c1", vector=[1.0], model="test")
        r = repr(emb)
        assert "test" in r


class TestEmbeddedChunk:
    """Tests for EmbeddedChunk."""

    def test_creation(self, sample_chunk: Chunk) -> None:
        """EmbeddedChunk should bind a chunk with an embedding."""
        emb = Embedding(chunk_id=sample_chunk.id, vector=[0.1, 0.2])
        ec = EmbeddedChunk(chunk=sample_chunk, embedding=emb)
        assert ec.chunk.id == sample_chunk.id
        assert ec.embedding.dimensions == 2


class TestSearchResult:
    """Tests for SearchResult and SearchResponse."""

    def test_search_result_repr(self, sample_chunk: Chunk) -> None:
        """SearchResult repr should include score."""
        sr = SearchResult(chunk=sample_chunk, score=0.95, rank=1)
        r = repr(sr)
        assert "0.95" in r

    def test_search_response_top_result(self, sample_chunk: Chunk) -> None:
        """top_result should return the first result."""
        sr = SearchResult(chunk=sample_chunk, score=0.9, rank=1)
        resp = SearchResponse(query="test", results=[sr], total_results=1)
        assert resp.top_result is sr

    def test_search_response_empty(self) -> None:
        """top_result should return None when empty."""
        resp = SearchResponse(query="test")
        assert resp.top_result is None

    def test_search_response_repr(self) -> None:
        """SearchResponse repr should include query."""
        resp = SearchResponse(query="hello", total_results=5)
        r = repr(resp)
        assert "hello" in r
