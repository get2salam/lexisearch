"""Tests for AdaptiveRetriever and supporting functions."""

from __future__ import annotations

import pytest

from lexisearch.retrieval.adaptive import (
    AdaptiveConfig,
    AdaptiveRetriever,
    _chunks_coverage,
    _query_keywords,
    _score_spread,
    adaptive_retriever,
)
from lexisearch.retrieval.advanced import AdvancedRetrievalResult, RetrievedChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(content: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=content[:8], content=content, score=score)


def _rich_retriever(query: str, top_k: int) -> list[RetrievedChunk]:
    """Returns top_k diverse chunks, each covering one unique term."""
    topics = [
        "consideration is the price of a promise",
        "contract requires offer and acceptance",
        "promissory estoppel prevents injustice",
        "void contracts lack legal effect",
        "damages remedy breach of contract",
        "equity provides injunctive relief",
        "fiduciary duty requires good faith",
        "tort liability arises from negligence",
    ]
    return [
        _chunk(topics[i % len(topics)], score=1.0 / (i + 1)) for i in range(min(top_k, len(topics)))
    ]


def _poor_retriever(query: str, top_k: int) -> list[RetrievedChunk]:
    """Returns chunks with irrelevant content (low coverage)."""
    return [_chunk("the weather is sunny and warm today", score=0.5) for _ in range(top_k)]


def _identical_score_retriever(query: str, top_k: int) -> list[RetrievedChunk]:
    """Returns chunks with identical scores."""
    return [_chunk(f"chunk content {i}", score=0.5) for i in range(top_k)]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestQueryKeywords:
    def test_extracts_meaningful_terms(self):
        kws = _query_keywords("What is the doctrine of promissory estoppel?")
        assert "doctrine" in kws
        assert "promissory" in kws
        assert "estoppel" in kws
        assert "is" not in kws

    def test_empty_query(self):
        assert _query_keywords("") == set()


class TestChunksCoverage:
    def test_full_coverage(self):
        chunks = [_chunk("consideration is a key concept in contract law")]
        kws = {"consideration", "contract"}
        assert _chunks_coverage(chunks, kws) == 1.0

    def test_zero_coverage(self):
        chunks = [_chunk("weather today is sunny and warm")]
        kws = {"consideration", "estoppel"}
        assert _chunks_coverage(chunks, kws) == 0.0

    def test_partial_coverage(self):
        chunks = [_chunk("contract is important in law")]
        kws = {"contract", "estoppel"}
        assert _chunks_coverage(chunks, kws) == pytest.approx(0.5)

    def test_empty_query_kws(self):
        chunks = [_chunk("anything")]
        assert _chunks_coverage(chunks, set()) == 1.0

    def test_empty_chunks(self):
        assert _chunks_coverage([], {"consideration"}) == 0.0


class TestScoreSpread:
    def test_single_chunk(self):
        assert _score_spread([_chunk("a", 0.5)]) == 0.0

    def test_uniform_scores(self):
        chunks = [_chunk(f"c{i}", 0.5) for i in range(5)]
        assert _score_spread(chunks) == 0.0

    def test_spread_computed(self):
        chunks = [_chunk("a", 0.9), _chunk("b", 0.5), _chunk("c", 0.1)]
        assert _score_spread(chunks) == pytest.approx(0.8)

    def test_empty(self):
        assert _score_spread([]) == 0.0


# ---------------------------------------------------------------------------
# AdaptiveRetriever
# ---------------------------------------------------------------------------


class TestAdaptiveRetriever:
    def setup_method(self):
        self.cfg = AdaptiveConfig(initial_k=2, max_k=8, min_coverage=0.5)

    def test_returns_advanced_retrieval_result(self):
        retriever = AdaptiveRetriever(_rich_retriever, config=self.cfg)
        result = retriever.retrieve("What is consideration in contract law?")
        assert isinstance(result, AdvancedRetrievalResult)

    def test_strategy_is_adaptive(self):
        retriever = AdaptiveRetriever(_rich_retriever, config=self.cfg)
        result = retriever.retrieve("consideration contract")
        assert result.strategy == "adaptive"

    def test_adaptive_metadata_present(self):
        retriever = AdaptiveRetriever(_rich_retriever, config=self.cfg)
        result = retriever.retrieve("consideration contract")
        assert "adaptive" in result.metadata
        meta = result.metadata["adaptive"]
        assert "final_k" in meta
        assert "final_coverage" in meta
        assert "expansion_steps" in meta
        assert "history" in meta

    def test_expansion_steps_recorded(self):
        retriever = AdaptiveRetriever(_rich_retriever, config=self.cfg)
        result = retriever.retrieve("consideration contract promissory estoppel")
        steps = result.metadata["adaptive"]["history"]
        assert len(steps) >= 1

    def test_expansion_steps_are_dicts(self):
        retriever = AdaptiveRetriever(_rich_retriever, config=self.cfg)
        result = retriever.retrieve("contract law")
        for step in result.metadata["adaptive"]["history"]:
            assert "k" in step
            assert "coverage" in step
            assert "reason" in step

    def test_top_k_override(self):
        retriever = AdaptiveRetriever(_rich_retriever, config=self.cfg)
        result = retriever.retrieve("consideration contract", top_k=2)
        assert len(result.chunks) <= 2

    def test_return_raw_returns_list(self):
        retriever = AdaptiveRetriever(_rich_retriever, config=self.cfg)
        raw = retriever.retrieve("consideration", return_raw=True)
        assert isinstance(raw, list)

    def test_poor_coverage_expands_k(self):
        """With min_coverage=0.5 and irrelevant chunks, should expand k."""
        cfg = AdaptiveConfig(initial_k=2, max_k=8, min_coverage=0.9, expand_factor=2.0)
        retriever = AdaptiveRetriever(_poor_retriever, config=cfg)
        result = retriever.retrieve("consideration contract promissory estoppel")
        meta = result.metadata["adaptive"]
        # Should have expanded at least once (or hit max_k)
        assert meta["expansion_steps"] >= 1

    def test_does_not_exceed_max_k(self):
        cfg = AdaptiveConfig(initial_k=2, max_k=5, min_coverage=1.0, expand_factor=2.0)
        retriever = AdaptiveRetriever(_poor_retriever, config=cfg)
        result = retriever.retrieve("consideration contract estoppel damages fiduciary")
        meta = result.metadata["adaptive"]
        assert meta["final_k"] <= 5

    def test_high_coverage_stops_early(self):
        """When initial k already covers the query, no expansion needed."""
        cfg = AdaptiveConfig(initial_k=3, max_k=20, min_coverage=0.1)
        retriever = AdaptiveRetriever(_rich_retriever, config=cfg)
        result = retriever.retrieve("contract")
        meta = result.metadata["adaptive"]
        # Only 1 expansion step (initial fetch, coverage met immediately)
        assert meta["expansion_steps"] == 1

    def test_low_spread_stops_expansion(self):
        """Uniform scores signal the index can't discriminate — stop early."""
        cfg = AdaptiveConfig(
            initial_k=2,
            max_k=8,
            min_coverage=1.0,  # extremely high — would normally keep expanding
            score_spread_threshold=0.1,
            expand_factor=2.0,
        )
        retriever = AdaptiveRetriever(_identical_score_retriever, config=cfg)
        result = retriever.retrieve("consideration contract estoppel")
        meta = result.metadata["adaptive"]
        # Should stop due to low spread after first expansion
        last_reason = meta["history"][-1]["reason"]
        assert last_reason in ("low_spread", "max_k", "coverage_met")

    def test_query_preserved_in_result(self):
        retriever = AdaptiveRetriever(_rich_retriever, config=self.cfg)
        result = retriever.retrieve("What is consideration?")
        assert result.query == "What is consideration?"

    def test_chunks_have_content(self):
        retriever = AdaptiveRetriever(_rich_retriever, config=self.cfg)
        result = retriever.retrieve("consideration contract")
        for chunk in result.chunks:
            assert chunk.content.strip()


# ---------------------------------------------------------------------------
# adaptive_retriever factory
# ---------------------------------------------------------------------------


class TestAdaptiveRetrieverFactory:
    def test_returns_adaptive_retriever(self):
        r = adaptive_retriever(_rich_retriever, initial_k=3, max_k=15)
        assert isinstance(r, AdaptiveRetriever)

    def test_config_applied(self):
        r = adaptive_retriever(_rich_retriever, initial_k=5, max_k=30, min_coverage=0.8)
        assert r.config.initial_k == 5
        assert r.config.max_k == 30
        assert r.config.min_coverage == pytest.approx(0.8)

    def test_retrieve_works(self):
        r = adaptive_retriever(_rich_retriever, initial_k=2, max_k=10)
        result = r.retrieve("contract law")
        assert isinstance(result, AdvancedRetrievalResult)
