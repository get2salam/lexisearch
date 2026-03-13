"""Tests for RetrievalExplainer and explanation data types."""

from __future__ import annotations

import pytest

from lexisearch.retrieval.explain import (
    ChunkExplanation,
    RetrievalExplanation,
    RetrievalExplainer,
    SubQueryContribution,
    TermOverlap,
    _keywords,
)
from lexisearch.retrieval.advanced import (
    AdvancedRetrievalConfig,
    AdvancedRetrievalResult,
    MultiQueryRetriever,
    RetrievedChunk,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, content: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, content=content, score=score)


def _result(
    query: str,
    chunks: list[RetrievedChunk],
    strategy: str = "multi_query",
    sub_queries: list[str] | None = None,
) -> AdvancedRetrievalResult:
    return AdvancedRetrievalResult(
        query=query,
        chunks=chunks,
        strategy=strategy,
        sub_queries=sub_queries or [query],
    )


# ---------------------------------------------------------------------------
# _keywords utility
# ---------------------------------------------------------------------------


class TestKeywords:
    def test_removes_stopwords(self):
        kws = _keywords("What is the consideration in a contract?")
        assert "is" not in kws
        assert "the" not in kws
        assert "consideration" in kws
        assert "contract" in kws

    def test_lowercases(self):
        kws = _keywords("Contract Law")
        assert "contract" in kws
        assert "Contract" not in kws

    def test_empty_string(self):
        assert _keywords("") == set()

    def test_all_stopwords(self):
        assert _keywords("is the a an") == set()


# ---------------------------------------------------------------------------
# RetrievalExplainer.explain
# ---------------------------------------------------------------------------


class TestRetrievalExplainer:
    def setup_method(self):
        self.explainer = RetrievalExplainer()
        self.query = "What is consideration in contract law?"
        self.chunks = [
            _chunk("c1", "Consideration is the price of a promise in contract law.", 0.9),
            _chunk("c2", "The weather today is sunny and warm.", 0.3),
            _chunk("c3", "Offer and acceptance form the basis of a valid contract.", 0.6),
        ]
        self.result = _result(self.query, self.chunks, sub_queries=[self.query, "consideration contract"])

    def test_returns_retrieval_explanation(self):
        exp = self.explainer.explain(self.result)
        assert isinstance(exp, RetrievalExplanation)

    def test_query_preserved(self):
        exp = self.explainer.explain(self.result)
        assert exp.query == self.query

    def test_strategy_preserved(self):
        exp = self.explainer.explain(self.result)
        assert exp.strategy == "multi_query"

    def test_correct_number_of_chunk_explanations(self):
        exp = self.explainer.explain(self.result)
        assert len(exp.chunks) == 3

    def test_chunk_ids_preserved(self):
        exp = self.explainer.explain(self.result)
        ids = [c.chunk_id for c in exp.chunks]
        assert ids == ["c1", "c2", "c3"]

    def test_final_scores_preserved(self):
        exp = self.explainer.explain(self.result)
        assert exp.chunks[0].final_score == pytest.approx(0.9)
        assert exp.chunks[1].final_score == pytest.approx(0.3)

    def test_term_overlap_computed(self):
        exp = self.explainer.explain(self.result)
        c1_exp = exp.chunks[0]
        assert isinstance(c1_exp.term_overlap, TermOverlap)
        # "consideration" and "contract" should both match c1
        assert "consideration" in c1_exp.term_overlap.matched_terms
        assert "contract" in c1_exp.term_overlap.matched_terms

    def test_irrelevant_chunk_low_overlap(self):
        exp = self.explainer.explain(self.result)
        c2_exp = exp.chunks[1]  # weather sentence
        assert c2_exp.term_overlap.overlap_ratio < 0.3

    def test_relevant_chunk_high_overlap(self):
        exp = self.explainer.explain(self.result)
        c1_exp = exp.chunks[0]  # consideration sentence
        assert c1_exp.term_overlap.overlap_ratio > 0.3

    def test_content_preview_truncated(self):
        long_content = "x" * 500
        result = _result(self.query, [_chunk("c1", long_content)])
        exp = self.explainer.explain(result)
        assert len(exp.chunks[0].content_preview) <= 200

    def test_empty_chunks(self):
        result = _result(self.query, [])
        exp = self.explainer.explain(result)
        assert exp.chunks == []

    def test_sub_queries_preserved(self):
        exp = self.explainer.explain(self.result)
        assert len(exp.sub_queries) == 2
        assert self.query in exp.sub_queries


# ---------------------------------------------------------------------------
# Sub-query contribution attribution
# ---------------------------------------------------------------------------


class TestSubQueryContributions:
    def setup_method(self):
        self.explainer = RetrievalExplainer(rrf_k=60)
        self.query = "contract consideration"
        self.chunk = _chunk("c1", "Consideration is the price of a promise.", 0.5)

    def test_contributions_computed_when_sub_results_provided(self):
        sub_results = [
            [_chunk("c1", "Consideration is the price of a promise."), _chunk("c2", "other")],
            [_chunk("c2", "other"), _chunk("c1", "Consideration is the price of a promise.")],
        ]
        result = _result(self.query, [self.chunk], sub_queries=["contract", "consideration"])
        exp = self.explainer.explain(result, sub_query_results=sub_results)
        c_exp = exp.chunks[0]
        assert len(c_exp.sub_query_contributions) == 2

    def test_rrf_contribution_for_rank_1(self):
        sub_results = [[self.chunk]]
        result = _result(self.query, [self.chunk], sub_queries=["contract"])
        exp = self.explainer.explain(result, sub_query_results=sub_results)
        contrib = exp.chunks[0].sub_query_contributions[0]
        assert contrib.rank == 1
        assert contrib.rrf_contribution == pytest.approx(1.0 / 61)

    def test_zero_contribution_when_not_retrieved(self):
        sub_results = [[_chunk("other", "different content")]]
        result = _result(self.query, [self.chunk], sub_queries=["contract"])
        exp = self.explainer.explain(result, sub_query_results=sub_results)
        contrib = exp.chunks[0].sub_query_contributions[0]
        assert contrib.rank == 0
        assert contrib.rrf_contribution == 0.0

    def test_no_contributions_without_sub_results(self):
        result = _result(self.query, [self.chunk])
        exp = self.explainer.explain(result)
        assert exp.chunks[0].sub_query_contributions == []


# ---------------------------------------------------------------------------
# explain_score (spot-check API)
# ---------------------------------------------------------------------------


class TestExplainScore:
    def setup_method(self):
        self.explainer = RetrievalExplainer()

    def test_returns_chunk_explanation(self):
        exp = self.explainer.explain_score(
            "What is consideration?",
            "Consideration is the price of a promise.",
            score=0.85,
        )
        assert isinstance(exp, ChunkExplanation)

    def test_score_preserved(self):
        exp = self.explainer.explain_score("contract law", "contract content", 0.77)
        assert exp.final_score == pytest.approx(0.77)

    def test_term_overlap_computed(self):
        exp = self.explainer.explain_score("consideration contract", "Consideration in contract.", 0.5)
        assert "consideration" in exp.term_overlap.matched_terms
        assert "contract" in exp.term_overlap.matched_terms


# ---------------------------------------------------------------------------
# Summary and rationale formatting
# ---------------------------------------------------------------------------


class TestFormattingMethods:
    def setup_method(self):
        self.explainer = RetrievalExplainer()
        self.query = "What is consideration in contract law?"
        chunks = [_chunk("c1", "Consideration is a core principle of contract law.", 0.9)]
        self.result = _result(self.query, chunks, sub_queries=[self.query])

    def test_chunk_summary_contains_chunk_id(self):
        exp = self.explainer.explain(self.result)
        summary = exp.chunks[0].summary()
        assert "c1" in summary

    def test_chunk_summary_contains_score(self):
        exp = self.explainer.explain(self.result)
        summary = exp.chunks[0].summary()
        assert "score=" in summary

    def test_chunk_rationale_contains_strategy(self):
        exp = self.explainer.explain(self.result)
        rationale = exp.chunks[0].rationale()
        assert "Strategy:" in rationale
        assert "multi_query" in rationale

    def test_chunk_rationale_contains_term_info(self):
        exp = self.explainer.explain(self.result)
        rationale = exp.chunks[0].rationale()
        assert "Term overlap:" in rationale
        assert "Matched" in rationale

    def test_retrieval_explanation_summary(self):
        exp = self.explainer.explain(self.result)
        summary = exp.summary()
        assert self.query in summary
        assert "Strategy:" in summary
