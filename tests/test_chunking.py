"""Tests for lexisearch.chunking strategies."""

from __future__ import annotations

import pytest

from lexisearch.chunking.fixed import FixedSizeChunker
from lexisearch.chunking.recursive import RecursiveChunker
from lexisearch.chunking.sentence import SentenceChunker
from lexisearch.chunking.semantic import SemanticChunker
from lexisearch.models import ChunkStrategy, Document


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_basic_chunking(self, sample_document: Document) -> None:
        """Should produce multiple chunks from a long document."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(sample_document)
        assert len(chunks) > 1

    def test_chunk_size_respected(self, sample_document: Document) -> None:
        """Each chunk should not exceed chunk_size."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(sample_document)
        for c in chunks:
            assert len(c.content) <= 100

    def test_empty_document(self, empty_document: Document) -> None:
        """Empty document should produce no chunks."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        assert chunker.chunk(empty_document) == []

    def test_short_document_single_chunk(self, short_document: Document) -> None:
        """A short document should produce exactly one chunk."""
        chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=0)
        chunks = chunker.chunk(short_document)
        assert len(chunks) == 1
        assert chunks[0].content == short_document.content

    def test_overlap_present(self) -> None:
        """Consecutive chunks should share overlapping text."""
        doc = Document(content="a" * 200)
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        # Second chunk should start with characters from end of first
        assert chunks[1].content[:20] == chunks[0].content[-20:]

    def test_strategy_is_fixed(self) -> None:
        """Strategy should be FIXED_SIZE."""
        chunker = FixedSizeChunker()
        assert chunker.strategy() == ChunkStrategy.FIXED_SIZE

    def test_invalid_chunk_size(self) -> None:
        """chunk_size <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=0)

    def test_overlap_exceeds_size(self) -> None:
        """overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_indices_sequential(self, sample_document: Document) -> None:
        """Chunk indices should be sequential starting from 0."""
        chunker = FixedSizeChunker(chunk_size=80, chunk_overlap=0)
        chunks = chunker.chunk(sample_document)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunk_document_id(self, sample_document: Document) -> None:
        """Every chunk should reference the source document."""
        chunker = FixedSizeChunker(chunk_size=80, chunk_overlap=0)
        chunks = chunker.chunk(sample_document)
        for c in chunks:
            assert c.document_id == sample_document.id


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_basic_chunking(self, sample_document: Document) -> None:
        """Should split into chunks."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(sample_document)
        assert len(chunks) >= 1

    def test_respects_paragraphs(self) -> None:
        """Should prefer splitting on paragraph boundaries."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        doc = Document(content=text)
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        # Should split between paragraphs, not mid-word
        assert len(chunks) >= 2

    def test_empty_document(self, empty_document: Document) -> None:
        """Empty document should produce no chunks."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        assert chunker.chunk(empty_document) == []

    def test_strategy_is_recursive(self) -> None:
        """Strategy should be RECURSIVE."""
        chunker = RecursiveChunker()
        assert chunker.strategy() == ChunkStrategy.RECURSIVE

    def test_chunk_many(self, sample_document: Document, short_document: Document) -> None:
        """chunk_many should process multiple documents."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk_many([sample_document, short_document])
        doc_ids = {c.document_id for c in chunks}
        assert sample_document.id in doc_ids
        assert short_document.id in doc_ids


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_basic_chunking(self, sample_document: Document) -> None:
        """Should produce chunks along sentence boundaries."""
        chunker = SentenceChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(sample_document)
        assert len(chunks) >= 1

    def test_empty_document(self, empty_document: Document) -> None:
        """Empty document should produce no chunks."""
        chunker = SentenceChunker(chunk_size=200, chunk_overlap=0)
        assert chunker.chunk(empty_document) == []

    def test_single_sentence(self) -> None:
        """A single sentence should produce one chunk."""
        doc = Document(content="This is a single complete sentence.")
        chunker = SentenceChunker(chunk_size=1000, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_strategy_is_sentence(self) -> None:
        """Strategy should be SENTENCE."""
        chunker = SentenceChunker()
        assert chunker.strategy() == ChunkStrategy.SENTENCE


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_basic_chunking(self, sample_document: Document) -> None:
        """Should produce chunks based on similarity."""
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=0, similarity_threshold=0.1)
        chunks = chunker.chunk(sample_document)
        assert len(chunks) >= 1

    def test_empty_document(self, empty_document: Document) -> None:
        """Empty document should produce no chunks."""
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=0)
        assert chunker.chunk(empty_document) == []

    def test_custom_similarity_fn(self) -> None:
        """A custom similarity function should be used."""
        calls: list[tuple[str, str]] = []

        def track_sim(a: str, b: str) -> float:
            calls.append((a, b))
            return 1.0  # Always similar

        doc = Document(content="Sentence one. Sentence two. Sentence three.")
        chunker = SemanticChunker(
            chunk_size=500,
            chunk_overlap=0,
            similarity_fn=track_sim,
        )
        chunker.chunk(doc)
        assert len(calls) > 0

    def test_strategy_is_semantic(self) -> None:
        """Strategy should be SEMANTIC."""
        chunker = SemanticChunker()
        assert chunker.strategy() == ChunkStrategy.SEMANTIC
