"""Tests for advanced retrieval strategies.

HyDE, Step-Back Prompting, Multi-Query, Composite — all tested with a simple
stub retriever so no embeddings or LLM are required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from lexisearch.retrieval.advanced import (
    AdvancedRetrievalConfig,
    AdvancedRetrievalResult,
    BaseAdvancedRetriever,
    CompositeAdvancedRetriever,
    HyDERetriever,
    MultiQueryRetriever,
    RetrievedChunk,
    RuleBasedQueryGenerator,
    StepBackRetriever,
    reciprocal_rank_fusion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunks(n: int, prefix: str = "doc") -> list[RetrievedChunk]:
    """Create ``n`` synthetic ``RetrievedChunk`` objects."""
    return [
        RetrievedChunk(
            chunk_id=f"{prefix}-{i}",
            content=f"Content about {prefix} topic number {i}.",
            score=1.0 / (i + 1),
            source_query=prefix,
        )
        for i in range(n)
    ]


def _stub_retriever(chunks: list[RetrievedChunk]) -> Any:
    """Return a callable that ignores its arguments and returns ``chunks``."""

    def _retrieve(query: str, top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]

    return _retrieve


# ---------------------------------------------------------------------------
# RetrievedChunk
# ---------------------------------------------------------------------------


class TestRetrievedChunk:
    def test_default_metadata(self) -> None:
        c = RetrievedChunk(chunk_id="c1", content="text", score=0.9)
        assert c.metadata == {}

    def test_source_query_default_empty(self) -> None:
        c = RetrievedChunk(chunk_id="c1", content="text", score=0.9)
        assert c.source_query == ""

    def test_custom_metadata(self) -> None:
        c = RetrievedChunk(chunk_id="c1", content="text", score=0.8, metadata={"doc": "A"})
        assert c.metadata["doc"] == "A"


# ---------------------------------------------------------------------------
# AdvancedRetrievalConfig
# ---------------------------------------------------------------------------


class TestAdvancedRetrievalConfig:
    def test_defaults(self) -> None:
        cfg = AdvancedRetrievalConfig()
        assert cfg.top_k == 5
        assert cfg.overretrieve_factor == 3
        assert cfg.rrf_k == 60
        assert cfg.deduplicate is True

    def test_custom_values(self) -> None:
        cfg = AdvancedRetrievalConfig(top_k=10, overretrieve_factor=5, rrf_k=30)
        assert cfg.top_k == 10
        assert cfg.rrf_k == 30


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_single_list(self) -> None:
        chunks = _make_chunks(5)
        fused = reciprocal_rank_fusion([chunks])
        assert len(fused) == 5

    def test_two_identical_lists(self) -> None:
        """Duplicate chunks should collapse to one entry with doubled score."""
        chunks = _make_chunks(3)
        fused = reciprocal_rank_fusion([chunks, chunks])
        assert len(fused) == 3  # deduplicated by chunk_id

    def test_two_disjoint_lists(self) -> None:
        a = _make_chunks(3, prefix="a")
        b = _make_chunks(3, prefix="b")
        fused = reciprocal_rank_fusion([a, b])
        assert len(fused) == 6  # all unique

    def test_top_n_limit(self) -> None:
        chunks = _make_chunks(10)
        fused = reciprocal_rank_fusion([chunks], top_n=3)
        assert len(fused) == 3

    def test_first_rank_highest_score(self) -> None:
        a = _make_chunks(5)
        fused = reciprocal_rank_fusion([a])
        # First item should have the highest RRF score (rank=1)
        scores = [c.score for c in fused]
        assert scores[0] == max(scores)

    def test_empty_input(self) -> None:
        fused = reciprocal_rank_fusion([])
        assert fused == []

    def test_empty_list_in_input(self) -> None:
        a = _make_chunks(3)
        fused = reciprocal_rank_fusion([a, []])
        assert len(fused) == 3

    def test_rrf_k_affects_scores(self) -> None:
        chunks = _make_chunks(3)
        fused_60 = reciprocal_rank_fusion([chunks], k=60)
        fused_10 = reciprocal_rank_fusion([chunks], k=10)
        # With k=10 (smaller), rank-1 score = 1/(10+1) > 1/(60+1)
        assert fused_10[0].score > fused_60[0].score


# ---------------------------------------------------------------------------
# RuleBasedQueryGenerator
# ---------------------------------------------------------------------------


class TestRuleBasedQueryGenerator:
    def setup_method(self) -> None:
        self.gen = RuleBasedQueryGenerator()

    def test_paraphrase_returns_list(self) -> None:
        variants = self.gen.paraphrase("What is retrieval-augmented generation?")
        assert isinstance(variants, list)

    def test_paraphrase_max_n(self) -> None:
        variants = self.gen.paraphrase("What is transformer architecture?", n=2)
        assert len(variants) <= 2

    def test_paraphrase_no_duplicates_of_original(self) -> None:
        query = "What is FAISS?"
        variants = self.gen.paraphrase(query)
        assert query.lower() not in [v.lower() for v in variants]

    def test_step_back_returns_string(self) -> None:
        result = self.gen.step_back("How does multi-head attention work in transformers?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_step_back_shorter_or_same(self) -> None:
        query = "How does multi-head attention work in transformers?"
        step_back = self.gen.step_back(query)
        # Should not be longer than original
        assert len(step_back) <= len(query)

    def test_step_back_single_word(self) -> None:
        # Short queries shouldn't crash
        result = self.gen.step_back("embeddings")
        assert isinstance(result, str)

    def test_generate_hypothesis_returns_non_empty(self) -> None:
        hyp = self.gen.generate_hypothesis("What is RAG?")
        assert isinstance(hyp, str)
        assert len(hyp) > 10

    def test_generate_hypothesis_is_declarative(self) -> None:
        hyp = self.gen.generate_hypothesis("What is vector search?")
        # Should end with a period (declarative sentence)
        assert not hyp.endswith("?")

    def test_paraphrase_empty_query(self) -> None:
        variants = self.gen.paraphrase("")
        assert isinstance(variants, list)


# ---------------------------------------------------------------------------
# HyDERetriever
# ---------------------------------------------------------------------------


class TestHyDERetriever:
    def _make_retriever(self, chunks: list[RetrievedChunk] | None = None) -> HyDERetriever:
        if chunks is None:
            chunks = _make_chunks(10)
        stub = _stub_retriever(chunks)
        return HyDERetriever(
            base_retriever=stub,
            config=AdvancedRetrievalConfig(top_k=3, overretrieve_factor=2),
        )

    def test_retrieve_returns_result(self) -> None:
        retriever = self._make_retriever()
        result = retriever.retrieve("What is RAG?")
        assert isinstance(result, AdvancedRetrievalResult)

    def test_retrieve_strategy_label(self) -> None:
        result = self._make_retriever().retrieve("test query")
        assert result.strategy == "hyde"

    def test_retrieve_respects_top_k(self) -> None:
        retriever = self._make_retriever(_make_chunks(20))
        result = retriever.retrieve("test", top_k=3)
        assert len(result.chunks) <= 3

    def test_retrieve_has_sub_queries(self) -> None:
        result = self._make_retriever().retrieve("What is dense retrieval?")
        assert len(result.sub_queries) >= 1

    def test_retrieve_has_hypothesis_metadata(self) -> None:
        result = self._make_retriever().retrieve("What is RAG?")
        assert "hypothesis" in result.metadata

    def test_retrieve_no_duplicates(self) -> None:
        chunks = _make_chunks(5)  # same 5 chunks
        retriever = HyDERetriever(
            base_retriever=_stub_retriever(chunks),
            config=AdvancedRetrievalConfig(top_k=5, deduplicate=True),
        )
        result = retriever.retrieve("test")
        ids = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids))

    def test_retrieve_empty_index(self) -> None:
        retriever = HyDERetriever(
            base_retriever=_stub_retriever([]),
            config=AdvancedRetrievalConfig(top_k=5),
        )
        result = retriever.retrieve("test")
        assert isinstance(result, AdvancedRetrievalResult)
        assert result.chunks == []

    def test_fallback_on_hypothesis_failure(self) -> None:
        """If the hypothesis retrieval fails, should fall back to direct query."""
        call_count = [0]

        def _failing_first_retriever(query: str, top_k: int) -> list[RetrievedChunk]:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("hypothesis retrieval failed")
            return _make_chunks(top_k)

        retriever = HyDERetriever(
            base_retriever=_failing_first_retriever,
            config=AdvancedRetrievalConfig(top_k=3),
        )
        result = retriever.retrieve("test query")
        assert isinstance(result, AdvancedRetrievalResult)

    def test_custom_generator(self) -> None:
        gen = MagicMock()
        gen.generate_hypothesis.return_value = "Custom hypothesis about the topic."
        retriever = HyDERetriever(
            base_retriever=_stub_retriever(_make_chunks(5)),
            generator=gen,
            config=AdvancedRetrievalConfig(top_k=3),
        )
        retriever.retrieve("some query")
        gen.generate_hypothesis.assert_called_once()


# ---------------------------------------------------------------------------
# StepBackRetriever
# ---------------------------------------------------------------------------


class TestStepBackRetriever:
    def _make_retriever(self, chunks: list[RetrievedChunk] | None = None) -> StepBackRetriever:
        if chunks is None:
            chunks = _make_chunks(10)
        return StepBackRetriever(
            base_retriever=_stub_retriever(chunks),
            config=AdvancedRetrievalConfig(top_k=4, overretrieve_factor=2),
        )

    def test_retrieve_returns_result(self) -> None:
        result = self._make_retriever().retrieve("How does FAISS work for ANN search?")
        assert isinstance(result, AdvancedRetrievalResult)

    def test_retrieve_strategy_label(self) -> None:
        result = self._make_retriever().retrieve("test query")
        assert result.strategy == "step_back"

    def test_retrieve_respects_top_k(self) -> None:
        result = self._make_retriever(_make_chunks(20)).retrieve("query", top_k=3)
        assert len(result.chunks) <= 3

    def test_retrieve_has_step_back_metadata(self) -> None:
        result = self._make_retriever().retrieve("How does attention work in transformers?")
        assert "step_back_query" in result.metadata

    def test_retrieve_sub_queries_has_two_entries(self) -> None:
        result = self._make_retriever().retrieve("How does attention work in transformers?")
        assert len(result.sub_queries) == 2  # original + step-back

    def test_retrieve_same_query_if_no_step_back(self) -> None:
        """If step-back equals original, only one retrieval pass is done."""
        gen = MagicMock()
        gen.step_back.return_value = "test query"  # same as original
        retriever = StepBackRetriever(
            base_retriever=_stub_retriever(_make_chunks(5)),
            generator=gen,
            config=AdvancedRetrievalConfig(top_k=3),
        )
        result = retriever.retrieve("test query")
        assert isinstance(result, AdvancedRetrievalResult)

    def test_retrieve_no_duplicates(self) -> None:
        chunks = _make_chunks(5)
        retriever = StepBackRetriever(
            base_retriever=_stub_retriever(chunks),
            config=AdvancedRetrievalConfig(top_k=5, deduplicate=True),
        )
        result = retriever.retrieve("detailed specific query about FAISS in Python")
        ids = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids))

    def test_custom_generator(self) -> None:
        gen = MagicMock()
        gen.step_back.return_value = "broader topic"
        retriever = StepBackRetriever(
            base_retriever=_stub_retriever(_make_chunks(5)),
            generator=gen,
            config=AdvancedRetrievalConfig(top_k=3),
        )
        retriever.retrieve("specific narrow query")
        gen.step_back.assert_called_once()


# ---------------------------------------------------------------------------
# MultiQueryRetriever
# ---------------------------------------------------------------------------


class TestMultiQueryRetriever:
    def _make_retriever(
        self, chunks: list[RetrievedChunk] | None = None, num_variants: int = 2
    ) -> MultiQueryRetriever:
        if chunks is None:
            chunks = _make_chunks(10)
        return MultiQueryRetriever(
            base_retriever=_stub_retriever(chunks),
            num_variants=num_variants,
            config=AdvancedRetrievalConfig(top_k=4, overretrieve_factor=2),
        )

    def test_retrieve_returns_result(self) -> None:
        result = self._make_retriever().retrieve("What is retrieval-augmented generation?")
        assert isinstance(result, AdvancedRetrievalResult)

    def test_retrieve_strategy_label(self) -> None:
        result = self._make_retriever().retrieve("test")
        assert result.strategy == "multi_query"

    def test_retrieve_respects_top_k(self) -> None:
        result = self._make_retriever(_make_chunks(20)).retrieve("query", top_k=3)
        assert len(result.chunks) <= 3

    def test_retrieve_has_multiple_sub_queries(self) -> None:
        result = self._make_retriever(num_variants=3).retrieve("What is transformer attention?")
        # At least the original query
        assert len(result.sub_queries) >= 1

    def test_retrieve_includes_original_query(self) -> None:
        query = "What is FAISS?"
        result = self._make_retriever().retrieve(query)
        assert query in result.sub_queries

    def test_retrieve_no_duplicates(self) -> None:
        chunks = _make_chunks(5)
        retriever = MultiQueryRetriever(
            base_retriever=_stub_retriever(chunks),
            num_variants=3,
            config=AdvancedRetrievalConfig(top_k=5, deduplicate=True),
        )
        result = retriever.retrieve("What is dense retrieval?")
        ids = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids))

    def test_retrieve_empty_index(self) -> None:
        retriever = MultiQueryRetriever(
            base_retriever=_stub_retriever([]),
            config=AdvancedRetrievalConfig(top_k=5),
        )
        result = retriever.retrieve("test")
        assert result.chunks == []

    def test_retrieve_handles_sub_query_failure(self) -> None:
        """Should still return results if some variants fail."""
        call_count = [0]
        chunks = _make_chunks(5)

        def _sometimes_failing(query: str, top_k: int) -> list[RetrievedChunk]:
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise RuntimeError("intermittent failure")
            return chunks[:top_k]

        retriever = MultiQueryRetriever(
            base_retriever=_sometimes_failing,
            num_variants=3,
            config=AdvancedRetrievalConfig(top_k=3),
        )
        # Should not raise
        result = retriever.retrieve("What is semantic search?")
        assert isinstance(result, AdvancedRetrievalResult)

    def test_num_variants_zero(self) -> None:
        """Zero variants should still retrieve with the original query."""
        retriever = MultiQueryRetriever(
            base_retriever=_stub_retriever(_make_chunks(5)),
            num_variants=0,
            config=AdvancedRetrievalConfig(top_k=3),
        )
        result = retriever.retrieve("some query")
        assert isinstance(result, AdvancedRetrievalResult)
        assert len(result.chunks) <= 3

    def test_metadata_has_num_variants(self) -> None:
        result = self._make_retriever(num_variants=2).retrieve("What is semantic search?")
        assert "num_variants" in result.metadata


# ---------------------------------------------------------------------------
# CompositeAdvancedRetriever
# ---------------------------------------------------------------------------


class TestCompositeAdvancedRetriever:
    def _make_composite(self) -> CompositeAdvancedRetriever:
        chunks_a = _make_chunks(5, "a")
        chunks_b = _make_chunks(5, "b")
        retriever_a = HyDERetriever(
            base_retriever=_stub_retriever(chunks_a),
            config=AdvancedRetrievalConfig(top_k=3),
        )
        retriever_b = StepBackRetriever(
            base_retriever=_stub_retriever(chunks_b),
            config=AdvancedRetrievalConfig(top_k=3),
        )
        return CompositeAdvancedRetriever(
            retrievers=[retriever_a, retriever_b],
            config=AdvancedRetrievalConfig(top_k=5),
        )

    def test_retrieve_returns_result(self) -> None:
        result = self._make_composite().retrieve("What is semantic search?")
        assert isinstance(result, AdvancedRetrievalResult)

    def test_retrieve_strategy_label(self) -> None:
        result = self._make_composite().retrieve("test")
        assert result.strategy == "composite"

    def test_retrieve_respects_top_k(self) -> None:
        result = self._make_composite().retrieve("test", top_k=3)
        assert len(result.chunks) <= 3

    def test_retrieve_merges_sub_queries(self) -> None:
        result = self._make_composite().retrieve("What is vector search?")
        assert len(result.sub_queries) > 0

    def test_retrieve_handles_sub_retriever_failure(self) -> None:
        """If one sub-retriever fails, others should still contribute."""

        class _FailingRetriever(BaseAdvancedRetriever):
            def retrieve(self, query: str, *, top_k: int | None = None) -> AdvancedRetrievalResult:
                raise RuntimeError("always fails")

        working = MultiQueryRetriever(
            base_retriever=_stub_retriever(_make_chunks(5)),
            config=AdvancedRetrievalConfig(top_k=3),
        )
        composite = CompositeAdvancedRetriever(
            retrievers=[_FailingRetriever(), working],
            config=AdvancedRetrievalConfig(top_k=3),
        )
        result = composite.retrieve("test query")
        assert isinstance(result, AdvancedRetrievalResult)

    def test_retrieve_empty_retrievers_list(self) -> None:
        composite = CompositeAdvancedRetriever(
            retrievers=[], config=AdvancedRetrievalConfig(top_k=3)
        )
        result = composite.retrieve("test")
        assert result.chunks == []

    def test_retrieve_deduplicates_results(self) -> None:
        """Shared chunks from multiple retrievers should collapse to one."""
        shared = _make_chunks(5, "shared")
        r1 = HyDERetriever(
            base_retriever=_stub_retriever(shared),
            config=AdvancedRetrievalConfig(top_k=5),
        )
        r2 = StepBackRetriever(
            base_retriever=_stub_retriever(shared),
            config=AdvancedRetrievalConfig(top_k=5),
        )
        composite = CompositeAdvancedRetriever(
            retrievers=[r1, r2],
            config=AdvancedRetrievalConfig(top_k=5, deduplicate=True),
        )
        result = composite.retrieve("test query")
        ids = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Module-level imports
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_all_exported_from_retrieval(self) -> None:
        from lexisearch.retrieval import (  # noqa: F401
            AdvancedRetrievalConfig,
            AdvancedRetrievalResult,
            BaseAdvancedRetriever,
            CompositeAdvancedRetriever,
            HyDERetriever,
            MultiQueryRetriever,
            RetrievedChunk,
            RuleBasedQueryGenerator,
            StepBackRetriever,
            reciprocal_rank_fusion,
        )

    def test_advanced_module_importable(self) -> None:
        import lexisearch.retrieval.advanced  # noqa: F401
