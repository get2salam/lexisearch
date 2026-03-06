"""Tests for multi-tenant namespace vector store wrapper.

Covers:
- Namespace qualification / stripping of chunk IDs
- Isolation: tenants cannot see each other's data
- add / upsert / delete / get / count / list_ids / search
- clear() selective and full
- Edge cases: empty namespace (root), invalid namespace, empty add
- Oversampling in search still returns correct top_k
"""

from __future__ import annotations

import pytest

from lexisearch.models import Chunk, ChunkStrategy, EmbeddedChunk, Embedding
from lexisearch.vectorstore import InMemoryVectorStore, VectorStoreConfig
from lexisearch.vectorstore.namespace import (
    NamespacedVectorStore,
    _qualify,
    _strip_ns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(dim: int = 4) -> InMemoryVectorStore:
    config = VectorStoreConfig(dimensions=dim)
    store = InMemoryVectorStore(config=config)
    store.initialize()
    return store


def _make_chunk(chunk_id: str, doc_id: str = "doc-1", dim: int = 4) -> EmbeddedChunk:
    chunk = Chunk(
        content=f"Content of {chunk_id}",
        document_id=doc_id,
        index=0,
        start_char=0,
        end_char=20,
        metadata={"source": "test"},
        strategy=ChunkStrategy.FIXED_SIZE,
        id=chunk_id,
    )
    embedding = Embedding(
        chunk_id=chunk_id,
        vector=[0.1, 0.2, 0.3, 0.4],
        model="test-model",
    )
    return EmbeddedChunk(chunk=chunk, embedding=embedding)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestQualifyHelpers:
    def test_qualify_basic(self) -> None:
        assert _qualify("tenant-a", "chunk-1") == "tenant-a::chunk-1"

    def test_qualify_empty_namespace(self) -> None:
        assert _qualify("", "chunk-1") == "chunk-1"

    def test_strip_ns_removes_prefix(self) -> None:
        assert _strip_ns("tenant-a", "tenant-a::chunk-1") == "chunk-1"

    def test_strip_ns_no_match(self) -> None:
        # Different namespace — do not strip
        assert _strip_ns("tenant-b", "tenant-a::chunk-1") == "tenant-a::chunk-1"

    def test_strip_ns_empty_namespace(self) -> None:
        assert _strip_ns("", "chunk-1") == "chunk-1"


# ---------------------------------------------------------------------------
# NamespacedVectorStore construction
# ---------------------------------------------------------------------------


class TestNamespacedVectorStoreInit:
    def test_repr(self) -> None:
        ns = NamespacedVectorStore(_make_store(), "tenant-x")
        assert "tenant-x" in repr(ns)
        assert "InMemoryVectorStore" in repr(ns)

    def test_invalid_namespace_raises(self) -> None:
        with pytest.raises(ValueError, match="separator"):
            NamespacedVectorStore(_make_store(), "bad::ns")

    def test_empty_namespace_allowed(self) -> None:
        ns = NamespacedVectorStore(_make_store(), "")
        assert ns.namespace == ""

    def test_config_forwarded(self) -> None:
        base = _make_store(dim=8)
        ns = NamespacedVectorStore(base, "ns")
        assert ns.config.dimensions == 8

    def test_context_manager(self) -> None:
        with NamespacedVectorStore(_make_store(), "ctx") as ns:
            assert ns.namespace == "ctx"


# ---------------------------------------------------------------------------
# add / upsert
# ---------------------------------------------------------------------------


class TestNamespacedAdd:
    def test_add_returns_unqualified_ids(self) -> None:
        ns = NamespacedVectorStore(_make_store(), "t1")
        ids = ns.add([_make_chunk("c1"), _make_chunk("c2")])
        assert ids == ["c1", "c2"]

    def test_add_qualifies_ids_in_underlying_store(self) -> None:
        base = _make_store()
        ns = NamespacedVectorStore(base, "t1")
        ns.add([_make_chunk("c1")])
        assert "t1::c1" in base.list_ids()
        assert "c1" not in base.list_ids()

    def test_add_empty_list_is_noop(self) -> None:
        ns = NamespacedVectorStore(_make_store(), "t1")
        ids = ns.add([])
        assert ids == []

    def test_upsert_qualifies_ids(self) -> None:
        base = _make_store()
        ns = NamespacedVectorStore(base, "t1")
        ns.upsert([_make_chunk("c1")])
        assert "t1::c1" in base.list_ids()

    def test_upsert_returns_unqualified_ids(self) -> None:
        ns = NamespacedVectorStore(_make_store(), "t2")
        ids = ns.upsert([_make_chunk("u1")])
        assert ids == ["u1"]


# ---------------------------------------------------------------------------
# Isolation between namespaces
# ---------------------------------------------------------------------------


class TestNamespaceIsolation:
    def test_tenants_cannot_see_each_other(self) -> None:
        base = _make_store()
        ta = NamespacedVectorStore(base, "tenant-a")
        tb = NamespacedVectorStore(base, "tenant-b")

        ta.add([_make_chunk("shared-id")])
        tb.add([_make_chunk("shared-id")])

        ids_a = ta.list_ids()
        ids_b = tb.list_ids()

        assert ids_a == ["shared-id"]
        assert ids_b == ["shared-id"]
        # Underlying store has both qualified IDs
        assert "tenant-a::shared-id" in base.list_ids()
        assert "tenant-b::shared-id" in base.list_ids()

    def test_search_returns_only_own_namespace(self) -> None:
        base = _make_store()
        ta = NamespacedVectorStore(base, "tenant-a")
        tb = NamespacedVectorStore(base, "tenant-b")

        ta.add([_make_chunk("a-chunk")])
        tb.add([_make_chunk("b-chunk")])

        results_a = ta.search([0.1, 0.2, 0.3, 0.4], top_k=10)
        results_b = tb.search([0.1, 0.2, 0.3, 0.4], top_k=10)

        assert all(r.chunk.id == "a-chunk" for r in results_a)
        assert all(r.chunk.id == "b-chunk" for r in results_b)

    def test_delete_only_affects_own_namespace(self) -> None:
        base = _make_store()
        ta = NamespacedVectorStore(base, "tenant-a")
        tb = NamespacedVectorStore(base, "tenant-b")

        ta.add([_make_chunk("shared")])
        tb.add([_make_chunk("shared")])

        deleted = ta.delete(["shared"])
        assert deleted == 1
        assert ta.count() == 0
        assert tb.count() == 1

    def test_count_is_namespace_scoped(self) -> None:
        base = _make_store()
        ta = NamespacedVectorStore(base, "tenant-a")
        tb = NamespacedVectorStore(base, "tenant-b")

        ta.add([_make_chunk("a1"), _make_chunk("a2")])
        tb.add([_make_chunk("b1")])

        assert ta.count() == 2
        assert tb.count() == 1
        assert base.count() == 3  # global count

    def test_list_ids_is_namespace_scoped(self) -> None:
        base = _make_store()
        ta = NamespacedVectorStore(base, "tenant-a")
        tb = NamespacedVectorStore(base, "tenant-b")

        ta.add([_make_chunk("x"), _make_chunk("y")])
        tb.add([_make_chunk("z")])

        assert sorted(ta.list_ids()) == ["x", "y"]
        assert tb.list_ids() == ["z"]


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestNamespacedGet:
    def test_get_by_unqualified_id(self) -> None:
        ns = NamespacedVectorStore(_make_store(), "t1")
        ns.add([_make_chunk("c1")])
        result = ns.get("c1")
        assert result is not None

    def test_get_missing_returns_none(self) -> None:
        ns = NamespacedVectorStore(_make_store(), "t1")
        assert ns.get("nonexistent") is None

    def test_get_cross_namespace_returns_none(self) -> None:
        base = _make_store()
        ta = NamespacedVectorStore(base, "ta")
        tb = NamespacedVectorStore(base, "tb")
        ta.add([_make_chunk("shared")])
        # tb cannot fetch ta's chunk by unqualified id
        assert tb.get("shared") is None


# ---------------------------------------------------------------------------
# search correctness
# ---------------------------------------------------------------------------


class TestNamespacedSearch:
    def test_search_returns_correct_top_k(self) -> None:
        ns = NamespacedVectorStore(_make_store(), "ns")
        for i in range(10):
            ns.add([_make_chunk(f"c{i}")])

        results = ns.search([0.1, 0.2, 0.3, 0.4], top_k=3)
        assert len(results) <= 3

    def test_search_results_have_sequential_rank(self) -> None:
        ns = NamespacedVectorStore(_make_store(), "ns")
        ns.add([_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")])
        results = ns.search([0.1, 0.2, 0.3, 0.4], top_k=3)
        for i, r in enumerate(results, start=1):
            assert r.rank == i

    def test_search_result_ids_are_unqualified(self) -> None:
        ns = NamespacedVectorStore(_make_store(), "myns")
        ns.add([_make_chunk("chunk-a")])
        results = ns.search([0.1, 0.2, 0.3, 0.4], top_k=5)
        assert all("::" not in r.chunk.id for r in results)

    def test_empty_store_returns_empty_results(self) -> None:
        ns = NamespacedVectorStore(_make_store(), "empty-ns")
        results = ns.search([0.1, 0.2, 0.3, 0.4], top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestNamespacedClear:
    def test_clear_removes_only_own_namespace(self) -> None:
        base = _make_store()
        ta = NamespacedVectorStore(base, "tenant-a")
        tb = NamespacedVectorStore(base, "tenant-b")

        ta.add([_make_chunk("a1")])
        tb.add([_make_chunk("b1")])

        ta.clear()

        assert ta.count() == 0
        assert tb.count() == 1

    def test_clear_all_namespaces_wipes_everything(self) -> None:
        base = _make_store()
        ta = NamespacedVectorStore(base, "tenant-a")
        tb = NamespacedVectorStore(base, "tenant-b")

        ta.add([_make_chunk("a1")])
        tb.add([_make_chunk("b1")])

        ta.clear(all_namespaces=True)

        assert base.count() == 0


# ---------------------------------------------------------------------------
# empty / root namespace
# ---------------------------------------------------------------------------


class TestRootNamespace:
    def test_empty_namespace_no_prefix(self) -> None:
        base = _make_store()
        ns = NamespacedVectorStore(base, "")
        ns.add([_make_chunk("c1")])
        # The ID is NOT qualified
        assert "c1" in base.list_ids()

    def test_empty_namespace_search(self) -> None:
        ns = NamespacedVectorStore(_make_store(), "")
        ns.add([_make_chunk("plain-id")])
        results = ns.search([0.1, 0.2, 0.3, 0.4], top_k=1)
        assert len(results) == 1
        assert results[0].chunk.id == "plain-id"
