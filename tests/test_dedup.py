"""Tests for the document deduplication module."""

from __future__ import annotations

from lexisearch.models import Chunk, SearchResult
from lexisearch.retrieval.dedup import (
    DeduplicationPipeline,
    DedupResult,
    DedupStats,
    DuplicateGroup,
    ExactDeduplicator,
    MinHashDeduplicator,
    SimHashDeduplicator,
    _hamming_distance,
    _normalise,
    _shingles,
    _simhash,
    _token_shingles,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(text: str, score: float = 0.9, doc_id: str = "") -> SearchResult:
    """Create a SearchResult for testing."""
    return SearchResult(
        chunk=Chunk(
            content=text,
            document_id=doc_id or "test_doc",
            metadata={"doc_id": doc_id or text[:20]},
        ),
        score=score,
    )


def _make_results(*texts: str) -> list[SearchResult]:
    """Create multiple SearchResults from text strings."""
    return [_make_result(t, score=1.0 - i * 0.1) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_lowercase(self) -> None:
        assert _normalise("Hello WORLD") == "hello world"

    def test_collapse_whitespace(self) -> None:
        assert _normalise("hello   world\n\nfoo") == "hello world foo"

    def test_strip_punctuation(self) -> None:
        assert _normalise("hello, world!") == "hello world"

    def test_empty(self) -> None:
        assert _normalise("") == ""


class TestShingles:
    def test_basic(self) -> None:
        shingles = _shingles("abcde", k=3)
        assert shingles == {"abc", "bcd", "cde"}

    def test_short_text(self) -> None:
        assert _shingles("ab", k=3) == {"ab"}

    def test_empty(self) -> None:
        assert _shingles("", k=3) == set()


class TestTokenShingles:
    def test_basic(self) -> None:
        shingles = _token_shingles("the quick brown fox", k=2)
        assert "the quick" in shingles
        assert "quick brown" in shingles
        assert "brown fox" in shingles

    def test_single_word(self) -> None:
        assert _token_shingles("hello", k=2) == {"hello"}


class TestHammingDistance:
    def test_identical(self) -> None:
        assert _hamming_distance(0xFF, 0xFF) == 0

    def test_one_bit(self) -> None:
        assert _hamming_distance(0b1111, 0b1110) == 1

    def test_all_different(self) -> None:
        assert _hamming_distance(0b0000, 0b1111) == 4


class TestSimHash:
    def test_similar_texts_close(self) -> None:
        h1 = _simhash("the quick brown fox jumps over the lazy dog")
        h2 = _simhash("the quick brown fox jumped over the lazy dog")
        dist = _hamming_distance(h1, h2)
        assert dist <= 20  # Should be relatively close

    def test_different_texts_far(self) -> None:
        h1 = _simhash("the quick brown fox jumps over the lazy dog")
        h2 = _simhash("quantum mechanics is a fundamental theory in physics")
        dist = _hamming_distance(h1, h2)
        assert dist > 10  # Should be far apart


# ---------------------------------------------------------------------------
# Test ExactDeduplicator
# ---------------------------------------------------------------------------


class TestExactDeduplicator:
    def test_no_duplicates(self) -> None:
        results = _make_results("hello world", "foo bar", "baz qux")
        dedup = ExactDeduplicator()
        out = dedup.deduplicate(results)
        assert out.stats.total == 3
        assert out.stats.unique == 3
        assert out.stats.duplicates == 0
        assert len(out.deduplicated) == 3
        assert len(out.duplicate_groups) == 0

    def test_exact_duplicates(self) -> None:
        results = _make_results("hello world", "hello world", "foo bar")
        dedup = ExactDeduplicator()
        out = dedup.deduplicate(results)
        assert out.stats.total == 3
        assert out.stats.unique == 2
        assert out.stats.duplicates == 1
        assert len(out.deduplicated) == 2
        assert len(out.duplicate_groups) == 1
        assert out.duplicate_groups[0].canonical_index == 0
        assert out.duplicate_groups[0].duplicate_indices == [1]

    def test_whitespace_normalisation(self) -> None:
        results = _make_results("hello world", "hello   world", "  hello world  ")
        dedup = ExactDeduplicator()
        out = dedup.deduplicate(results)
        assert out.stats.unique == 1
        assert out.stats.duplicates == 2

    def test_case_normalisation(self) -> None:
        results = _make_results("Hello World", "HELLO WORLD", "hello world")
        dedup = ExactDeduplicator()
        out = dedup.deduplicate(results)
        assert out.stats.unique == 1

    def test_punctuation_normalisation(self) -> None:
        results = _make_results("hello, world!", "hello world", "hello; world.")
        dedup = ExactDeduplicator()
        out = dedup.deduplicate(results)
        assert out.stats.unique == 1

    def test_empty_input(self) -> None:
        dedup = ExactDeduplicator()
        out = dedup.deduplicate([])
        assert out.stats.total == 0
        assert out.stats.unique == 0
        assert out.stats.duplicates == 0
        assert len(out.deduplicated) == 0

    def test_single_result(self) -> None:
        results = _make_results("single item")
        dedup = ExactDeduplicator()
        out = dedup.deduplicate(results)
        assert out.stats.unique == 1

    def test_preserves_order(self) -> None:
        results = _make_results("aaa", "bbb", "aaa", "ccc")
        dedup = ExactDeduplicator()
        out = dedup.deduplicate(results)
        assert [r.chunk.content for r in out.deduplicated] == ["aaa", "bbb", "ccc"]

    def test_multiple_groups(self) -> None:
        results = _make_results("aaa", "bbb", "aaa", "bbb", "ccc")
        dedup = ExactDeduplicator()
        out = dedup.deduplicate(results)
        assert out.stats.unique == 3
        assert out.stats.duplicates == 2
        assert len(out.duplicate_groups) == 2

    def test_timing(self) -> None:
        results = _make_results("hello world")
        dedup = ExactDeduplicator()
        out = dedup.deduplicate(results)
        assert out.stats.dedup_time_ms >= 0


# ---------------------------------------------------------------------------
# Test SimHashDeduplicator
# ---------------------------------------------------------------------------


class TestSimHashDeduplicator:
    def test_identical_texts(self) -> None:
        results = _make_results("the cat sat on the mat", "the cat sat on the mat")
        dedup = SimHashDeduplicator(hamming_threshold=3)
        out = dedup.deduplicate(results)
        assert out.stats.duplicates == 1

    def test_very_similar_texts(self) -> None:
        results = _make_results(
            "the court held that the defendant was liable for damages",
            "the court held that the defendant was liable for damage",
        )
        dedup = SimHashDeduplicator(hamming_threshold=5)
        out = dedup.deduplicate(results)
        assert out.stats.duplicates >= 1

    def test_different_texts_kept(self) -> None:
        results = _make_results(
            "the court found the defendant guilty of fraud",
            "quantum mechanics describes behaviour of particles at atomic scale",
        )
        dedup = SimHashDeduplicator(hamming_threshold=3)
        out = dedup.deduplicate(results)
        assert out.stats.unique == 2

    def test_empty_input(self) -> None:
        dedup = SimHashDeduplicator()
        out = dedup.deduplicate([])
        assert out.stats.total == 0

    def test_strict_threshold(self) -> None:
        results = _make_results(
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumped over the lazy dog",
        )
        dedup = SimHashDeduplicator(hamming_threshold=0)
        out = dedup.deduplicate(results)
        assert out.stats.unique == 2  # Strict = no fuzzy match

    def test_groups_populated(self) -> None:
        results = _make_results("abc def ghi", "abc def ghi")
        dedup = SimHashDeduplicator(hamming_threshold=3)
        out = dedup.deduplicate(results)
        assert len(out.duplicate_groups) == 1
        assert out.duplicate_groups[0].canonical_index == 0


# ---------------------------------------------------------------------------
# Test MinHashDeduplicator
# ---------------------------------------------------------------------------


class TestMinHashDeduplicator:
    def test_identical_texts(self) -> None:
        results = _make_results(
            "the supreme court upheld the decision of the high court",
            "the supreme court upheld the decision of the high court",
        )
        dedup = MinHashDeduplicator(threshold=0.8)
        out = dedup.deduplicate(results)
        assert out.stats.duplicates == 1

    def test_similar_texts(self) -> None:
        results = _make_results(
            "the supreme court upheld the decision of the high court in this matter",
            "the supreme court upheld the decision of the high court in this case",
        )
        dedup = MinHashDeduplicator(threshold=0.5, num_perm=256)
        out = dedup.deduplicate(results)
        assert out.stats.duplicates >= 1

    def test_different_texts_kept(self) -> None:
        results = _make_results(
            "the supreme court upheld the decision regarding property rights",
            "quantum computing uses qubits to perform parallel calculations",
        )
        dedup = MinHashDeduplicator(threshold=0.5)
        out = dedup.deduplicate(results)
        assert out.stats.unique == 2

    def test_empty_input(self) -> None:
        dedup = MinHashDeduplicator()
        out = dedup.deduplicate([])
        assert out.stats.total == 0

    def test_high_threshold_strict(self) -> None:
        results = _make_results(
            "the court held that the appellant was liable",
            "the court held that the respondent was liable",
        )
        dedup = MinHashDeduplicator(threshold=0.99)
        out = dedup.deduplicate(results)
        assert out.stats.unique == 2  # Not similar enough at 99%


# ---------------------------------------------------------------------------
# Test DeduplicationPipeline
# ---------------------------------------------------------------------------


class TestDeduplicationPipeline:
    def test_empty_pipeline(self) -> None:
        results = _make_results("aaa", "bbb")
        pipeline = DeduplicationPipeline()
        out = pipeline.deduplicate(results)
        assert out.stats.unique == 2  # No strategies = no dedup

    def test_single_strategy(self) -> None:
        results = _make_results("aaa", "aaa", "bbb")
        pipeline = DeduplicationPipeline([ExactDeduplicator()])
        out = pipeline.deduplicate(results)
        assert out.stats.unique == 2

    def test_chained_strategies(self) -> None:
        results = _make_results(
            "the court held that the defendant was guilty",
            "the court held that the defendant was guilty",  # exact dup
            "the court held that the defendant was guilty of the crime",  # near dup
            "quantum mechanics is a theory of physics",  # different
        )
        pipeline = DeduplicationPipeline(
            [
                ExactDeduplicator(),
                SimHashDeduplicator(hamming_threshold=5),
            ]
        )
        out = pipeline.deduplicate(results)
        assert out.stats.total == 4
        assert out.stats.unique <= 3  # At least exact dup removed

    def test_add_method(self) -> None:
        pipeline = DeduplicationPipeline()
        pipeline.add(ExactDeduplicator()).add(SimHashDeduplicator())
        assert len(pipeline.strategies) == 2

    def test_stats_accuracy(self) -> None:
        results = _make_results("aaa", "aaa", "bbb", "bbb", "ccc")
        pipeline = DeduplicationPipeline([ExactDeduplicator()])
        out = pipeline.deduplicate(results)
        assert out.stats.total == 5
        assert out.stats.unique == 3
        assert out.stats.duplicates == 2
        assert out.stats.total == out.stats.unique + out.stats.duplicates

    def test_timing(self) -> None:
        results = _make_results("hello", "world")
        pipeline = DeduplicationPipeline([ExactDeduplicator()])
        out = pipeline.deduplicate(results)
        assert out.stats.dedup_time_ms >= 0

    def test_original_preserved(self) -> None:
        results = _make_results("aaa", "aaa", "bbb")
        pipeline = DeduplicationPipeline([ExactDeduplicator()])
        out = pipeline.deduplicate(results)
        assert len(out.original) == 3
        assert len(out.deduplicated) == 2


# ---------------------------------------------------------------------------
# Test DedupResult / DedupStats / DuplicateGroup dataclasses
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_dedup_stats(self) -> None:
        stats = DedupStats(total=10, unique=8, duplicates=2, dedup_time_ms=1.5)
        assert stats.total == 10
        assert stats.unique == 8
        assert stats.duplicates == 2
        assert stats.dedup_time_ms == 1.5

    def test_duplicate_group(self) -> None:
        group = DuplicateGroup(canonical_index=0, duplicate_indices=[1, 2], similarity=0.95)
        assert group.canonical_index == 0
        assert group.duplicate_indices == [1, 2]
        assert group.similarity == 0.95

    def test_duplicate_group_defaults(self) -> None:
        group = DuplicateGroup(canonical_index=5)
        assert group.duplicate_indices == []
        assert group.similarity == 1.0

    def test_dedup_result(self) -> None:
        result = DedupResult(
            original=[],
            deduplicated=[],
            duplicate_groups=[],
            stats=DedupStats(0, 0, 0, 0.0),
        )
        assert result.original == []
        assert result.deduplicated == []
