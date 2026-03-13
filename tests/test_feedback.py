"""Tests for the lexisearch.feedback package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from lexisearch.feedback import (
    DiskFeedbackStore,
    FeedbackRanker,
    FeedbackType,
    InMemoryFeedbackStore,
    RetrievalFeedback,
)
from lexisearch.models import Chunk, ChunkStrategy, SearchResult

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# RetrievalFeedback tests
# ---------------------------------------------------------------------------


class TestRetrievalFeedback:
    def test_thumbs_up_score(self) -> None:
        fb = RetrievalFeedback.thumbs_up("query", "c1")
        assert fb.score == 1.0
        assert fb.feedback_type == FeedbackType.THUMBS_UP

    def test_thumbs_down_score(self) -> None:
        fb = RetrievalFeedback.thumbs_down("query", "c1")
        assert fb.score == 0.0
        assert fb.feedback_type == FeedbackType.THUMBS_DOWN

    def test_rated_valid(self) -> None:
        fb = RetrievalFeedback.rated("query", "c1", 0.75)
        assert fb.score == 0.75
        assert fb.feedback_type == FeedbackType.RATING

    def test_rated_clamps_raises_on_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            RetrievalFeedback.rated("q", "c", 1.5)
        with pytest.raises(ValueError):
            RetrievalFeedback.rated("q", "c", -0.1)

    def test_clicked_score_decreases_with_rank(self) -> None:
        fb_rank0 = RetrievalFeedback.clicked("q", "c", rank=0)
        fb_rank3 = RetrievalFeedback.clicked("q", "c", rank=3)
        assert fb_rank0.score > fb_rank3.score

    def test_clicked_stores_rank_in_metadata(self) -> None:
        fb = RetrievalFeedback.clicked("q", "c", rank=2)
        assert fb.metadata["rank"] == 2

    def test_unique_ids(self) -> None:
        fb1 = RetrievalFeedback.thumbs_up("q", "c")
        fb2 = RetrievalFeedback.thumbs_up("q", "c")
        assert fb1.id != fb2.id

    def test_metadata_forwarded(self) -> None:
        fb = RetrievalFeedback.thumbs_up("q", "c", session="sess-1")
        assert fb.metadata["session"] == "sess-1"

    def test_repr(self) -> None:
        fb = RetrievalFeedback.thumbs_up("q", "c")
        r = repr(fb)
        assert "thumbs_up" in r
        assert "1.00" in r


# ---------------------------------------------------------------------------
# InMemoryFeedbackStore tests
# ---------------------------------------------------------------------------


class TestInMemoryFeedbackStore:
    def _fb(self, chunk_id: str, score: float = 1.0, query: str = "q") -> RetrievalFeedback:
        return RetrievalFeedback(
            query=query,
            chunk_id=chunk_id,
            feedback_type=FeedbackType.RATING,
            score=score,
        )

    def test_record_increases_count(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(self._fb("c1"))
        assert store.total_feedback() == 1

    def test_record_many(self) -> None:
        store = InMemoryFeedbackStore()
        store.record_many([self._fb("c1"), self._fb("c2"), self._fb("c3")])
        assert len(store) == 3

    def test_get_for_chunk(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(self._fb("c1"))
        store.record(self._fb("c2"))
        store.record(self._fb("c1"))
        assert len(store.get_for_chunk("c1")) == 2

    def test_get_for_query(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(self._fb("c1", query="q1"))
        store.record(self._fb("c2", query="q2"))
        store.record(self._fb("c3", query="q1"))
        assert len(store.get_for_query("q1")) == 2

    def test_aggregate_chunk_score_mean(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(self._fb("c1", score=1.0))
        store.record(self._fb("c1", score=0.0))
        assert store.aggregate_chunk_score("c1") == pytest.approx(0.5)

    def test_aggregate_chunk_score_none_when_empty(self) -> None:
        store = InMemoryFeedbackStore()
        assert store.aggregate_chunk_score("nonexistent") is None

    def test_aggregate_query_chunk_score(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(self._fb("c1", score=0.8, query="q1"))
        store.record(self._fb("c1", score=0.6, query="q1"))
        store.record(self._fb("c1", score=0.0, query="q2"))  # different query
        result = store.aggregate_query_chunk_score("q1", "c1")
        assert result == pytest.approx(0.7)

    def test_positive_rate(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(self._fb("c1", score=1.0))
        store.record(self._fb("c1", score=1.0))
        store.record(self._fb("c1", score=0.0))
        rate = store.positive_rate("c1")
        assert rate == pytest.approx(2 / 3)

    def test_positive_rate_none_when_empty(self) -> None:
        store = InMemoryFeedbackStore()
        assert store.positive_rate("nonexistent") is None

    def test_clear(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(self._fb("c1"))
        store.clear()
        assert len(store) == 0

    def test_to_dicts_and_from_dicts(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(RetrievalFeedback.thumbs_up("q1", "c1"))
        store.record(RetrievalFeedback.thumbs_down("q2", "c2"))
        dicts = store.to_dicts()
        assert len(dicts) == 2
        restored = InMemoryFeedbackStore.from_dicts(dicts)
        assert len(restored) == 2
        assert restored.aggregate_chunk_score("c1") == 1.0

    def test_repr(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(self._fb("c1"))
        assert "events=1" in repr(store)


# ---------------------------------------------------------------------------
# DiskFeedbackStore tests
# ---------------------------------------------------------------------------


class TestDiskFeedbackStore:
    def test_persists_and_reloads(self, tmp_dir: Path) -> None:
        path = tmp_dir / "feedback.json"
        store = DiskFeedbackStore(path)
        store.record(RetrievalFeedback.thumbs_up("q", "c1"))
        store.record(RetrievalFeedback.thumbs_down("q", "c2"))

        # Reload from disk
        store2 = DiskFeedbackStore(path)
        assert store2.total_feedback() == 2
        assert store2.aggregate_chunk_score("c1") == 1.0
        assert store2.aggregate_chunk_score("c2") == 0.0

    def test_record_many_saves_once(self, tmp_dir: Path) -> None:
        path = tmp_dir / "batch.json"
        store = DiskFeedbackStore(path)
        fbs = [RetrievalFeedback.thumbs_up(f"q{i}", f"c{i}") for i in range(5)]
        store.record_many(fbs)
        store2 = DiskFeedbackStore(path)
        assert len(store2) == 5

    def test_corrupt_file_starts_fresh(self, tmp_dir: Path) -> None:
        path = tmp_dir / "corrupt.json"
        path.write_text("NOT VALID JSON", encoding="utf-8")
        store = DiskFeedbackStore(path)  # should not raise
        assert len(store) == 0

    def test_repr_includes_path(self, tmp_dir: Path) -> None:
        path = tmp_dir / "f.json"
        store = DiskFeedbackStore(path)
        assert "f.json" in repr(store)


# ---------------------------------------------------------------------------
# FeedbackRanker tests
# ---------------------------------------------------------------------------


def _make_result(chunk_id: str, score: float, rank: int = 1) -> SearchResult:
    chunk = Chunk(
        content="Test content",
        document_id="doc-1",
        index=0,
        strategy=ChunkStrategy.FIXED_SIZE,
    )
    # Override id for deterministic tests
    object.__setattr__(chunk, "id", chunk_id)
    return SearchResult(chunk=chunk, score=score, rank=rank)


class TestFeedbackRanker:
    def test_boost_applied_to_positive_chunk(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(RetrievalFeedback.thumbs_up("q", "c1"))
        ranker = FeedbackRanker(store, boost=0.2)
        results = [_make_result("c1", 0.5)]
        reranked = ranker.rerank(results, query="q")
        assert reranked[0].score == pytest.approx(0.7)

    def test_penalty_applied_to_negative_chunk(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(RetrievalFeedback.thumbs_down("q", "c1"))
        ranker = FeedbackRanker(store, penalty=0.3)
        results = [_make_result("c1", 0.8)]
        reranked = ranker.rerank(results, query="q")
        assert reranked[0].score == pytest.approx(0.5)

    def test_no_feedback_unchanged(self) -> None:
        store = InMemoryFeedbackStore()
        ranker = FeedbackRanker(store)
        results = [_make_result("c1", 0.7)]
        reranked = ranker.rerank(results)
        assert reranked[0].score == pytest.approx(0.7)

    def test_score_clamped_to_max_1(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(RetrievalFeedback.thumbs_up("q", "c1"))
        ranker = FeedbackRanker(store, boost=0.5)
        results = [_make_result("c1", 0.9)]
        reranked = ranker.rerank(results, query="q")
        assert reranked[0].score <= 1.0

    def test_score_clamped_to_min_0(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(RetrievalFeedback.thumbs_down("q", "c1"))
        ranker = FeedbackRanker(store, penalty=0.9)
        results = [_make_result("c1", 0.1)]
        reranked = ranker.rerank(results, query="q")
        assert reranked[0].score >= 0.0

    def test_reordering_by_adjusted_score(self) -> None:
        store = InMemoryFeedbackStore()
        # c2 gets boost, c1 stays the same
        store.record(RetrievalFeedback.thumbs_up("q", "c2"))
        ranker = FeedbackRanker(store, boost=0.3)
        results = [_make_result("c1", 0.8, rank=1), _make_result("c2", 0.6, rank=2)]
        reranked = ranker.rerank(results, query="q")
        # c2 (0.6 + 0.3 = 0.9) should now rank 1
        assert reranked[0].chunk.id == "c2"
        assert reranked[1].chunk.id == "c1"

    def test_ranks_updated_after_reorder(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(RetrievalFeedback.thumbs_up("q", "c2"))
        ranker = FeedbackRanker(store, boost=0.3)
        results = [_make_result("c1", 0.8, rank=1), _make_result("c2", 0.6, rank=2)]
        reranked = ranker.rerank(results, query="q")
        assert reranked[0].rank == 1
        assert reranked[1].rank == 2

    def test_feedback_adjusted_flag_set(self) -> None:
        store = InMemoryFeedbackStore()
        store.record(RetrievalFeedback.thumbs_up("q", "c1"))
        ranker = FeedbackRanker(store)
        results = [_make_result("c1", 0.5), _make_result("c2", 0.4)]
        reranked = ranker.rerank(results, query="q")
        c1_result = next(r for r in reranked if r.chunk.id == "c1")
        c2_result = next(r for r in reranked if r.chunk.id == "c2")
        assert c1_result.metadata["feedback_adjusted"] is True
        assert c2_result.metadata["feedback_adjusted"] is False

    def test_query_aware_uses_per_query_score(self) -> None:
        store = InMemoryFeedbackStore()
        # Good feedback for query "q1", bad for "q2"
        store.record(RetrievalFeedback.rated("q1", "c1", rating=1.0))
        store.record(RetrievalFeedback.rated("q2", "c1", rating=0.0))
        ranker = FeedbackRanker(store, boost=0.2, penalty=0.3, query_aware=True)
        # For q1 the chunk should be boosted
        r_q1 = ranker.rerank([_make_result("c1", 0.5)], query="q1")
        assert r_q1[0].score == pytest.approx(0.7)
        # For q2 the chunk should be penalised
        r_q2 = ranker.rerank([_make_result("c1", 0.5)], query="q2")
        assert r_q2[0].score == pytest.approx(0.2)

    def test_empty_results_returns_empty(self) -> None:
        store = InMemoryFeedbackStore()
        ranker = FeedbackRanker(store)
        assert ranker.rerank([]) == []
