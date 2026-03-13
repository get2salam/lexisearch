"""Tests for contextual compression (SentenceCompressor and KeywordCompressor)."""

from __future__ import annotations

from typing import ClassVar

from lexisearch.retrieval.advanced import RetrievedChunk
from lexisearch.retrieval.compression import (
    BaseCompressor,
    CompressedChunk,
    KeywordCompressor,
    SentenceCompressor,
    _query_keywords,
    _sentence_score,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, content=content, score=1.0)


_LEGAL_TEXT = (
    "A contract requires offer, acceptance, and consideration. "
    "The weather today is sunny with mild winds. "
    "Consideration must be sufficient but need not be adequate. "
    "The restaurant serves Italian food on weekends."
)

_QUERY = "What is consideration in contract law?"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestQueryKeywords:
    def test_removes_stop_words(self):
        kws = _query_keywords("What is the meaning of res judicata?")
        # stop words must be absent
        assert "is" not in kws
        assert "the" not in kws
        assert "of" not in kws
        # content words must be present
        assert "meaning" in kws
        assert "res" in kws

    def test_empty_query(self):
        assert _query_keywords("") == set()

    def test_all_stop_words(self):
        assert _query_keywords("is the a an") == set()


class TestSentenceScore:
    def test_high_overlap(self):
        score = _sentence_score(
            "Consideration must be sufficient but need not be adequate.",
            {"consideration", "sufficient", "adequate"},
        )
        assert score > 0.2

    def test_no_overlap(self):
        score = _sentence_score(
            "The weather today is sunny.",
            {"consideration", "contract"},
        )
        # Length bonus yields a small positive value; ensure score is near zero
        assert score < 0.1

    def test_empty_query_keywords(self):
        assert _sentence_score("Any sentence.", set()) == 0.0

    def test_score_in_range(self):
        kws = {"offer", "acceptance", "contract"}
        score = _sentence_score("Offer and acceptance form a valid contract.", kws)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# SentenceCompressor
# ---------------------------------------------------------------------------


class TestSentenceCompressor:
    def setup_method(self):
        self.compressor = SentenceCompressor(threshold=0.05, min_sentences=1)

    def test_returns_compressed_chunks(self):
        chunks = [_chunk("c1", _LEGAL_TEXT)]
        results = self.compressor.compress(_QUERY, chunks)
        assert len(results) == 1
        assert isinstance(results[0], CompressedChunk)

    def test_relevant_sentences_retained(self):
        chunks = [_chunk("c1", _LEGAL_TEXT)]
        result = self.compressor.compress(_QUERY, chunks)[0]
        # Consideration-related sentences should be retained
        assert "consideration" in result.compressed_content.lower()

    def test_irrelevant_sentences_dropped(self):
        chunks = [_chunk("c1", _LEGAL_TEXT)]
        result = self.compressor.compress(_QUERY, chunks)[0]
        # Weather and restaurant sentences should be dropped
        assert "weather" not in result.compressed_content.lower()
        assert "restaurant" not in result.compressed_content.lower()

    def test_compression_ratio_between_0_and_1(self):
        chunks = [_chunk("c1", _LEGAL_TEXT)]
        result = self.compressor.compress(_QUERY, chunks)[0]
        assert 0.0 < result.compression_ratio <= 1.0

    def test_fallback_when_nothing_passes_threshold(self):
        """With a very high threshold, fallback to min_sentences best."""
        compressor = SentenceCompressor(threshold=0.99, min_sentences=1)
        chunks = [_chunk("c1", _LEGAL_TEXT)]
        result = compressor.compress(_QUERY, chunks)[0]
        # Must still return something
        assert result.compressed_content.strip()
        assert len(result.retained_sentences) >= 1

    def test_max_sentences_respected(self):
        compressor = SentenceCompressor(threshold=0.0, min_sentences=1, max_sentences=2)
        chunks = [_chunk("c1", _LEGAL_TEXT)]
        result = compressor.compress(_QUERY, chunks)[0]
        assert len(result.retained_sentences) <= 2

    def test_empty_content_chunk(self):
        chunks = [_chunk("empty", "")]
        results = self.compressor.compress(_QUERY, chunks)
        assert len(results) == 1

    def test_multiple_chunks(self):
        chunks = [_chunk(f"c{i}", _LEGAL_TEXT) for i in range(5)]
        results = self.compressor.compress(_QUERY, chunks)
        assert len(results) == 5

    def test_chunk_id_preserved(self):
        chunks = [_chunk("my-id-123", _LEGAL_TEXT)]
        result = self.compressor.compress(_QUERY, chunks)[0]
        assert result.chunk_id == "my-id-123"

    def test_original_content_preserved(self):
        chunks = [_chunk("c1", _LEGAL_TEXT)]
        result = self.compressor.compress(_QUERY, chunks)[0]
        assert result.original_content == _LEGAL_TEXT

    def test_relevance_score_range(self):
        chunks = [_chunk("c1", _LEGAL_TEXT)]
        result = self.compressor.compress(_QUERY, chunks)[0]
        assert 0.0 <= result.relevance_score <= 1.0

    def test_is_base_compressor(self):
        assert isinstance(self.compressor, BaseCompressor)


# ---------------------------------------------------------------------------
# KeywordCompressor
# ---------------------------------------------------------------------------


class TestKeywordCompressor:
    def setup_method(self):
        self.compressor = KeywordCompressor(min_keyword_matches=1, min_sentences=1)

    def test_returns_compressed_chunks(self):
        chunks = [_chunk("c1", _LEGAL_TEXT)]
        results = self.compressor.compress(_QUERY, chunks)
        assert len(results) == 1

    def test_keyword_sentences_retained(self):
        result = self.compressor.compress(_QUERY, [_chunk("c1", _LEGAL_TEXT)])[0]
        assert "consideration" in result.compressed_content.lower()

    def test_non_keyword_sentences_dropped(self):
        result = self.compressor.compress(_QUERY, [_chunk("c1", _LEGAL_TEXT)])[0]
        assert "weather" not in result.compressed_content.lower()

    def test_min_sentences_fallback(self):
        """When no keywords match, must still return at least min_sentences."""
        compressor = KeywordCompressor(min_keyword_matches=100, min_sentences=1)
        result = compressor.compress("xyzzy plugh", [_chunk("c1", _LEGAL_TEXT)])[0]
        assert result.retained_sentences

    def test_empty_query(self):
        result = self.compressor.compress("", [_chunk("c1", _LEGAL_TEXT)])[0]
        # Should return something (min_sentences fallback)
        assert result.compressed_content

    def test_metadata_preserved(self):
        chunk = RetrievedChunk(
            chunk_id="c1", content=_LEGAL_TEXT, score=0.8, metadata={"source": "test"}
        )
        result = self.compressor.compress(_QUERY, [chunk])[0]
        assert result.metadata == {"source": "test"}

    def test_is_base_compressor(self):
        assert isinstance(self.compressor, BaseCompressor)


# ---------------------------------------------------------------------------
# Compression with plain dicts / duck-typing
# ---------------------------------------------------------------------------


class TestDuckTypingSupport:
    """Compressors should work with any object that has .chunk_id and .content."""

    def test_works_with_duck_typed_objects(self):
        class FakeChunk:
            chunk_id = "fake-1"
            content = "Consideration is the price of a promise in contract law."
            metadata: ClassVar[dict[str, str]] = {}

        compressor = SentenceCompressor(threshold=0.0, min_sentences=1)
        results = compressor.compress("What is consideration?", [FakeChunk()])
        assert len(results) == 1
        assert results[0].chunk_id == "fake-1"
