"""Tests for content-hash deduplication (lexisearch.ingest.dedup).

Covers:
- content_hash() stability and normalisation
- ContentHashRegistry.register() / contains() / reset()
- DeduplicationFilter.filter() with all three strategies
- DeduplicationFilter.load() and load_many() with a mock loader
- DeduplicationFilter.stats()
- Thread safety (basic)
- Edge cases: empty list, single doc, all duplicates
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from lexisearch.ingest.dedup import (
    ContentHashRegistry,
    DeduplicationFilter,
    DuplicateDocumentError,
    _normalise,
    content_hash,
)
from lexisearch.models import Document, DocumentMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(content: str, doc_id: str = "d1", title: str = "Test Doc") -> Document:
    return Document(content=content, id=doc_id, metadata=DocumentMetadata(title=title))


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_same_content_same_hash(self) -> None:
        d1 = _doc("Hello world")
        d2 = _doc("Hello world")
        assert content_hash(d1) == content_hash(d2)

    def test_different_content_different_hash(self) -> None:
        d1 = _doc("Hello world")
        d2 = _doc("Goodbye world")
        assert content_hash(d1) != content_hash(d2)

    def test_whitespace_normalised(self) -> None:
        d1 = _doc("Hello   world")
        d2 = _doc("Hello world")
        assert content_hash(d1) == content_hash(d2)

    def test_case_normalised(self) -> None:
        d1 = _doc("HELLO WORLD")
        d2 = _doc("hello world")
        assert content_hash(d1) == content_hash(d2)

    def test_leading_trailing_whitespace_normalised(self) -> None:
        d1 = _doc("  hello world  ")
        d2 = _doc("hello world")
        assert content_hash(d1) == content_hash(d2)

    def test_returns_64_char_hex(self) -> None:
        digest = content_hash(_doc("anything"))
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_newlines_normalised(self) -> None:
        d1 = _doc("line one\nline two")
        d2 = _doc("line one line two")
        assert content_hash(d1) == content_hash(d2)


# ---------------------------------------------------------------------------
# _normalise helper
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_strips_edges(self) -> None:
        assert _normalise("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self) -> None:
        assert _normalise("a   b") == "a b"

    def test_lowercases(self) -> None:
        assert _normalise("ABC") == "abc"


# ---------------------------------------------------------------------------
# ContentHashRegistry
# ---------------------------------------------------------------------------


class TestContentHashRegistry:
    def test_register_new_returns_true(self) -> None:
        reg = ContentHashRegistry()
        is_new, _ = reg.register(_doc("unique content"))
        assert is_new is True

    def test_register_duplicate_returns_false(self) -> None:
        reg = ContentHashRegistry()
        reg.register(_doc("same content"))
        is_new, _ = reg.register(_doc("same content"))
        assert is_new is False

    def test_seen_count_increments(self) -> None:
        reg = ContentHashRegistry()
        reg.register(_doc("a"))
        reg.register(_doc("b"))
        assert reg.seen_count() == 2

    def test_seen_count_no_double_count(self) -> None:
        reg = ContentHashRegistry()
        reg.register(_doc("same"))
        reg.register(_doc("same"))
        assert reg.seen_count() == 1

    def test_reset_clears_registry(self) -> None:
        reg = ContentHashRegistry()
        reg.register(_doc("something"))
        reg.reset()
        assert reg.seen_count() == 0
        is_new, _ = reg.register(_doc("something"))
        assert is_new is True

    def test_contains_true_after_register(self) -> None:
        reg = ContentHashRegistry()
        doc = _doc("check me")
        reg.register(doc)
        assert reg.contains(doc) is True

    def test_contains_false_before_register(self) -> None:
        reg = ContentHashRegistry()
        assert reg.contains(_doc("not yet")) is False

    def test_add_digest_manually(self) -> None:
        reg = ContentHashRegistry()
        doc = _doc("manual digest")
        digest = content_hash(doc)
        reg.add_digest(digest)
        is_new, _ = reg.register(doc)
        assert is_new is False

    def test_shared_registry_across_filters(self) -> None:
        shared = ContentHashRegistry()
        f1 = DeduplicationFilter(registry=shared)
        f2 = DeduplicationFilter(registry=shared)

        f1.filter([_doc("shared content")])
        # f2 uses the same registry so sees it as duplicate
        result = f2.filter([_doc("shared content")])
        assert result == []


# ---------------------------------------------------------------------------
# DeduplicationFilter.filter()
# ---------------------------------------------------------------------------


class TestDeduplicationFilterFilter:
    def test_unique_documents_pass_through(self) -> None:
        f = DeduplicationFilter()
        docs = [_doc("doc a", "d1"), _doc("doc b", "d2")]
        result = f.filter(docs)
        assert len(result) == 2

    def test_duplicate_removed(self) -> None:
        f = DeduplicationFilter()
        docs = [_doc("same content"), _doc("same content")]
        result = f.filter(docs)
        assert len(result) == 1

    def test_first_occurrence_kept(self) -> None:
        f = DeduplicationFilter()
        d1 = _doc("identical", "id-1")
        d2 = _doc("identical", "id-2")
        result = f.filter([d1, d2])
        assert result[0].id == "id-1"

    def test_empty_input_returns_empty(self) -> None:
        f = DeduplicationFilter()
        assert f.filter([]) == []

    def test_all_duplicates_returns_empty(self) -> None:
        f = DeduplicationFilter()
        docs = [_doc("same"), _doc("same"), _doc("same")]
        result = f.filter(docs)
        assert len(result) == 1  # first occurrence kept

    def test_multiple_calls_accumulate_registry(self) -> None:
        f = DeduplicationFilter()
        f.filter([_doc("seen before")])
        result = f.filter([_doc("seen before")])
        assert result == []

    def test_strategy_warn_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        f = DeduplicationFilter(strategy="warn")
        with caplog.at_level(logging.WARNING, logger="lexisearch.ingest.dedup"):
            f.filter([_doc("dup"), _doc("dup")])
        assert any("Duplicate" in r.message for r in caplog.records)

    def test_strategy_raise_on_duplicate(self) -> None:
        f = DeduplicationFilter(strategy="raise")
        with pytest.raises(DuplicateDocumentError) as exc_info:
            f.filter([_doc("same"), _doc("same")])
        assert exc_info.value.digest != ""
        assert exc_info.value.document is not None

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid dedup strategy"):
            DeduplicationFilter(strategy="invalid")


# ---------------------------------------------------------------------------
# DeduplicationFilter.stats()
# ---------------------------------------------------------------------------


class TestDeduplicationFilterStats:
    def test_stats_initial(self) -> None:
        f = DeduplicationFilter()
        s = f.stats()
        assert s == {
            "total_seen": 0,
            "duplicates_skipped": 0,
            "unique_loaded": 0,
            "registry_size": 0,
        }

    def test_stats_after_filter(self) -> None:
        f = DeduplicationFilter()
        f.filter([_doc("a"), _doc("b"), _doc("a")])
        s = f.stats()
        assert s["total_seen"] == 3
        assert s["duplicates_skipped"] == 1
        assert s["unique_loaded"] == 2
        assert s["registry_size"] == 2

    def test_reset_clears_stats(self) -> None:
        f = DeduplicationFilter()
        f.filter([_doc("x")])
        f.reset()
        s = f.stats()
        assert s["total_seen"] == 0
        assert s["registry_size"] == 0


# ---------------------------------------------------------------------------
# DeduplicationFilter.load() with a mock loader
# ---------------------------------------------------------------------------


class TestDeduplicationFilterLoad:
    def _make_loader(self, doc: Document) -> MagicMock:
        loader = MagicMock()
        loader.load.return_value = doc
        return loader

    def _make_list_loader(self, docs: list[Document]) -> MagicMock:
        loader = MagicMock()
        loader.load.return_value = docs
        return loader

    def test_load_new_document(self) -> None:
        doc = _doc("fresh content")
        f = DeduplicationFilter(self._make_loader(doc))
        result = f.load("file.txt")
        assert result == doc

    def test_load_returns_original_on_duplicate(self) -> None:
        doc = _doc("seen before")
        loader = self._make_loader(doc)
        f = DeduplicationFilter(loader)
        f.load("file.txt")  # first time
        result = f.load("file.txt")  # second time — duplicate
        # The original doc is returned (caller decides what to do)
        assert result == doc

    def test_load_list(self) -> None:
        docs = [_doc("x", "d1"), _doc("y", "d2"), _doc("x", "d3")]
        f = DeduplicationFilter(self._make_list_loader(docs))
        result = f.load("dir/")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_load_without_loader_raises(self) -> None:
        f = DeduplicationFilter()
        with pytest.raises(RuntimeError, match="No loader"):
            f.load("file.txt")

    def test_load_many_deduplicates_across_files(self) -> None:
        doc_a = _doc("content A", "d1")
        doc_b = _doc("content B", "d2")
        doc_a2 = _doc("content A", "d3")  # duplicate of doc_a

        loader = MagicMock()
        loader.load.side_effect = [doc_a, doc_b, doc_a2]

        f = DeduplicationFilter(loader)
        results = f.load_many(["f1.txt", "f2.txt", "f3.txt"])
        assert len(results) == 2

    def test_load_many_without_loader_raises(self) -> None:
        f = DeduplicationFilter()
        with pytest.raises(RuntimeError):
            f.load_many(["a.txt"])


# ---------------------------------------------------------------------------
# is_duplicate
# ---------------------------------------------------------------------------


class TestIsDuplicate:
    def test_false_before_seeing(self) -> None:
        f = DeduplicationFilter()
        assert f.is_duplicate(_doc("new content")) is False

    def test_true_after_filtering(self) -> None:
        f = DeduplicationFilter()
        doc = _doc("will be seen")
        f.filter([doc])
        assert f.is_duplicate(doc) is True


# ---------------------------------------------------------------------------
# DuplicateDocumentError
# ---------------------------------------------------------------------------


class TestDuplicateDocumentError:
    def test_has_digest_and_document(self) -> None:
        doc = _doc("test")
        err = DuplicateDocumentError("msg", digest="abc123", document=doc)
        assert err.digest == "abc123"
        assert err.document is doc

    def test_is_exception(self) -> None:
        assert isinstance(DuplicateDocumentError("err"), Exception)


# ---------------------------------------------------------------------------
# Thread safety (basic)
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_register_no_double_registration(self) -> None:
        registry = ContentHashRegistry()
        doc = _doc("concurrent content")
        results: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            is_new, _ = registry.register(doc)
            with lock:
                results.append(is_new)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one thread should have seen it as new
        assert results.count(True) == 1
        assert results.count(False) == 19
