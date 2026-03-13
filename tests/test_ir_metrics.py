"""Tests for information-retrieval evaluation metrics."""

from __future__ import annotations

import math

import pytest

from lexisearch.evaluation.ir_metrics import (
    RetrievalEvalReport,
    average_precision,
    compute_ir_metrics,
    mean_average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RETRIEVED = ["doc1", "doc2", "doc3", "doc4", "doc5"]
RELEVANT_ALL = {"doc1", "doc2", "doc3", "doc4", "doc5"}
RELEVANT_NONE: set[str] = set()
RELEVANT_FIRST = {"doc1"}
RELEVANT_LAST = {"doc5"}
RELEVANT_MID = {"doc3"}


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(RETRIEVED, RELEVANT_FIRST, k=5) == 1.0

    def test_zero_recall_no_relevant_retrieved(self):
        assert recall_at_k(["a", "b", "c"], {"x", "y"}, k=3) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(RETRIEVED, RELEVANT_NONE, k=5) == 0.0

    def test_partial_recall(self):
        score = recall_at_k(["doc1", "doc2", "x", "y", "z"], {"doc1", "doc2", "doc3"}, k=5)
        assert pytest.approx(score) == 2 / 3

    def test_k_cutoff_respected(self):
        # relevant is only at position 5, k=3 cuts it off
        score = recall_at_k(RETRIEVED, RELEVANT_LAST, k=3)
        assert score == 0.0

    def test_k_default_uses_full_list(self):
        score = recall_at_k(RETRIEVED, RELEVANT_LAST)
        assert score == 1.0

    def test_all_relevant_all_retrieved(self):
        assert recall_at_k(RETRIEVED, RELEVANT_ALL, k=5) == 1.0

    def test_score_in_range(self):
        score = recall_at_k(RETRIEVED, {"doc2", "doc4"}, k=5)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"x", "y"}, k=3) == 0.0

    def test_half_relevant(self):
        score = precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4)
        assert pytest.approx(score) == 0.5

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a"}, k=5) == 0.0

    def test_k_cutoff(self):
        # 2 out of first 3 are relevant, even though more exist further down
        score = precision_at_k(["a", "b", "c", "d", "e"], {"a", "b"}, k=3)
        assert pytest.approx(score) == 2 / 3


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------


class TestAveragePrecision:
    def test_first_is_relevant(self):
        # P@1=1.0, AP = 1.0 / 1 = 1.0
        ap = average_precision(["a", "b", "c"], {"a"})
        assert pytest.approx(ap) == 1.0

    def test_last_is_relevant(self):
        # P@3=1/3, AP = (1/3) / 1 = 1/3
        ap = average_precision(["a", "b", "c"], {"c"})
        assert pytest.approx(ap) == 1 / 3

    def test_all_relevant(self):
        # P@1=1, P@2=1, P@3=1 → AP = (1+1+1)/3 = 1.0
        ap = average_precision(["a", "b", "c"], {"a", "b", "c"})
        assert pytest.approx(ap) == 1.0

    def test_no_relevant_returns_zero(self):
        ap = average_precision(["a", "b", "c"], {"x", "y"})
        assert ap == 0.0

    def test_empty_relevant_returns_zero(self):
        ap = average_precision(["a", "b"], set())
        assert ap == 0.0

    def test_interleaved_relevant(self):
        # Retrieved: [r, -, r, -, r]; relevant = {r0, r2, r4}
        retrieved = ["r0", "nr1", "r2", "nr3", "r4"]
        relevant = {"r0", "r2", "r4"}
        ap = average_precision(retrieved, relevant)
        # P@1=1, P@3=2/3, P@5=3/5 → sum=1+2/3+3/5=2.233…; AP=2.233/3=0.744
        assert 0.7 < ap < 0.8

    def test_k_cutoff_limits_scoring(self):
        ap_k3 = average_precision(["a", "b", "c", "d"], {"d"}, k=3)
        assert ap_k3 == 0.0  # relevant at pos 4, cutoff k=3

    def test_score_in_range(self):
        ap = average_precision(["doc1", "doc2", "doc3"], {"doc1", "doc3"})
        assert 0.0 <= ap <= 1.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNdcgAtK:
    def test_perfect_ranking(self):
        # Relevant docs in positions 1,2 → ideal
        score = ndcg_at_k(["r1", "r2", "nr1"], {"r1", "r2"}, k=3)
        assert pytest.approx(score) == 1.0

    def test_worst_ranking_binary(self):
        # Relevant docs only at positions 4,5 out of 5
        score = ndcg_at_k(["nr1", "nr2", "nr3", "r1", "r2"], {"r1", "r2"}, k=5)
        dcg = 1 / math.log2(5) + 1 / math.log2(6)
        ideal_dcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert pytest.approx(score) == dcg / ideal_dcg

    def test_empty_relevant_returns_zero(self):
        assert ndcg_at_k(["a", "b"], set()) == 0.0

    def test_graded_relevance(self):
        # Graded: doc1=3, doc2=2, doc3=1
        retrieved = ["doc2", "doc1", "doc3"]
        grades = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        score = ndcg_at_k(retrieved, grades, k=3)
        assert 0.0 < score < 1.0  # not perfect (doc1 should be first)

    def test_graded_perfect_order(self):
        retrieved = ["doc1", "doc2", "doc3"]
        grades = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        score = ndcg_at_k(retrieved, grades, k=3)
        assert pytest.approx(score) == 1.0

    def test_k_cutoff(self):
        score_k1 = ndcg_at_k(["r1", "nr1", "r2"], {"r1", "r2"}, k=1)
        assert pytest.approx(score_k1) == 1.0  # only r1 matters at k=1


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------


class TestReciprocalRank:
    def test_first_is_relevant(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_is_relevant(self):
        assert reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(1 / 2)

    def test_third_is_relevant(self):
        assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)

    def test_none_relevant_returns_zero(self):
        assert reciprocal_rank(["a", "b", "c"], {"x"}) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert reciprocal_rank(["a", "b"], set()) == 0.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


class TestMRR:
    def test_all_first(self):
        retrieved = [["a", "b"], ["c", "d"]]
        relevant = [{"a"}, {"c"}]
        assert mrr(retrieved, relevant) == 1.0

    def test_mixed(self):
        retrieved = [["a", "b", "c"], ["x", "y", "z"]]
        relevant = [{"b"}, {"z"}]
        expected = (1 / 2 + 1 / 3) / 2
        assert mrr(retrieved, relevant) == pytest.approx(expected)

    def test_none_found(self):
        retrieved = [["a", "b"], ["c", "d"]]
        relevant = [{"x"}, {"y"}]
        assert mrr(retrieved, relevant) == 0.0

    def test_empty_input(self):
        assert mrr([], []) == 0.0


# ---------------------------------------------------------------------------
# mean_average_precision
# ---------------------------------------------------------------------------


class TestMAP:
    def test_perfect_map(self):
        retrieved = [["a"], ["b"]]
        relevant = [{"a"}, {"b"}]
        assert mean_average_precision(retrieved, relevant) == 1.0

    def test_zero_map(self):
        retrieved = [["a", "b"], ["c", "d"]]
        relevant = [{"x"}, {"y"}]
        assert mean_average_precision(retrieved, relevant) == 0.0

    def test_empty_input(self):
        assert mean_average_precision([], []) == 0.0


# ---------------------------------------------------------------------------
# compute_ir_metrics (batch runner)
# ---------------------------------------------------------------------------


class TestComputeIRMetrics:
    def _run(self, retrieved, relevant, k=5):
        return compute_ir_metrics(retrieved, relevant, k=k)

    def test_returns_report(self):
        report = self._run([["a", "b", "c"]], [{"a"}])
        assert isinstance(report, RetrievalEvalReport)

    def test_num_queries_correct(self):
        queries = [["a"], ["b"], ["c"]]
        rels = [{"a"}, {"b"}, {"c"}]
        report = compute_ir_metrics(queries, rels, k=1)
        assert report.num_queries == 3

    def test_perfect_scores(self):
        retrieved = [["a", "b", "c"], ["d", "e", "f"]]
        relevant = [{"a", "b", "c"}, {"d", "e", "f"}]
        report = self._run(retrieved, relevant, k=3)
        assert report.recall_at_k == 1.0
        assert report.precision_at_k == 1.0
        assert report.mrr == 1.0
        assert report.map_score == 1.0
        assert report.ndcg_at_k == 1.0

    def test_empty_input(self):
        report = compute_ir_metrics([], [], k=5)
        assert report.num_queries == 0
        assert report.recall_at_k == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must have the same length"):
            compute_ir_metrics([["a"]], [{"a"}, {"b"}])

    def test_per_query_disabled_by_default(self):
        report = compute_ir_metrics([["a"]], [{"a"}], k=1)
        assert report.per_query == []

    def test_per_query_enabled(self):
        retrieved = [["a", "b"], ["c", "d"]]
        relevant = [{"a"}, {"d"}]
        report = compute_ir_metrics(retrieved, relevant, k=2, include_per_query=True)
        assert len(report.per_query) == 2
        for pq in report.per_query:
            assert "recall_at_k" in pq
            assert "precision_at_k" in pq
            assert "ap" in pq
            assert "rr" in pq
            assert "ndcg_at_k" in pq

    def test_summary_string(self):
        report = compute_ir_metrics([["a"]], [{"a"}], k=5)
        summary = report.summary()
        assert "MRR=" in summary
        assert "MAP=" in summary
        assert "NDCG@k=" in summary

    def test_k_stored_in_report(self):
        report = compute_ir_metrics([["a"]], [{"a"}], k=7)
        assert report.k == 7

    def test_scores_in_range(self):
        retrieved = [["d1", "d2", "d3", "d4", "d5"]] * 10
        relevant = [{"d1", "d3", "d5"}] * 10
        report = compute_ir_metrics(retrieved, relevant, k=5)
        for field_name in ("recall_at_k", "precision_at_k", "mrr", "map_score", "ndcg_at_k"):
            val = getattr(report, field_name)
            assert 0.0 <= val <= 1.0, f"{field_name}={val} out of range"
