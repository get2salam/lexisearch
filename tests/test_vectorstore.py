"""Comprehensive tests for the vector store layer.

Tests cover:
- Distance metric functions (cosine, euclidean, dot product)
- InMemoryVectorStore (full CRUD, search, persistence, filtering)
- BaseVectorStore interface contract
- Edge cases (empty stores, dimension mismatches, duplicate IDs)
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pytest

from lexisearch.embeddings.mock import MockEmbedder
from lexisearch.models import (
    Chunk,
    ChunkStrategy,
    EmbeddedChunk,
    Embedding,
    SearchResult,
)
from lexisearch.vectorstore.base import DistanceMetric, VectorStoreConfig
from lexisearch.vectorstore.memory import InMemoryVectorStore
from lexisearch.vectorstore.metrics import (
    compute_pairwise_scores,
    compute_score,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    l2_normalize,
)

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def embedder() -> MockEmbedder:
    """Return a mock embedder with 8 dimensions for fast tests."""
    return MockEmbedder(dimensions=8)


@pytest.fixture
def store_config() -> VectorStoreConfig:
    """Return a basic vector store config."""
    return VectorStoreConfig(
        collection_name="test_collection",
        dimensions=8,
        metric=DistanceMetric.COSINE,
    )


@pytest.fixture
def memory_store(store_config: VectorStoreConfig) -> InMemoryVectorStore:
    """Return an initialized InMemoryVectorStore."""
    store = InMemoryVectorStore(config=store_config)
    store.initialize()
    return store


def _make_embedded_chunk(
    content: str,
    vector: list[float],
    chunk_id: str | None = None,
    document_id: str = "doc-1",
    index: int = 0,
    metadata: dict[str, Any] | None = None,
) -> EmbeddedChunk:
    """Helper to create an EmbeddedChunk for testing."""
    chunk = Chunk(
        content=content,
        document_id=document_id,
        index=index,
        start_char=0,
        end_char=len(content),
        metadata=metadata or {},
        strategy=ChunkStrategy.FIXED_SIZE,
        id=chunk_id or f"chunk-{index}",
    )
    embedding = Embedding(
        chunk_id=chunk.id,
        vector=vector,
        model="test-model",
    )
    return EmbeddedChunk(chunk=chunk, embedding=embedding)


def _make_sample_items(dims: int = 8) -> list[EmbeddedChunk]:
    """Create a set of sample embedded chunks for testing."""
    embedder = MockEmbedder(dimensions=dims)
    texts = [
        "Machine learning algorithms process data",
        "Neural networks enable deep learning",
        "Natural language processing understands text",
        "Computer vision analyses images",
        "Reinforcement learning optimises decisions",
    ]
    items: list[EmbeddedChunk] = []
    for i, text in enumerate(texts):
        vector = embedder.embed_text(text)
        items.append(
            _make_embedded_chunk(
                content=text,
                vector=vector,
                chunk_id=f"chunk-{i}",
                document_id=f"doc-{i // 3}",
                index=i,
                metadata={"topic": "ai" if i < 3 else "other"},
            )
        )
    return items


# =====================================================================
# Metric function tests
# =====================================================================


class TestCosineSimiarity:
    """Tests for cosine_similarity."""

    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_unit_vectors(self) -> None:
        a = l2_normalize([3.0, 4.0])
        b = l2_normalize([4.0, 3.0])
        sim = cosine_similarity(a, b)
        assert 0.0 < sim < 1.0

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity([1.0, 2.0], [1.0])


class TestEuclideanDistance:
    """Tests for euclidean_distance."""

    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert euclidean_distance(v, v) == pytest.approx(0.0)

    def test_known_distance(self) -> None:
        a = [0.0, 0.0]
        b = [3.0, 4.0]
        assert euclidean_distance(a, b) == pytest.approx(5.0)

    def test_single_dimension(self) -> None:
        assert euclidean_distance([0.0], [5.0]) == pytest.approx(5.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            euclidean_distance([1.0], [1.0, 2.0])


class TestDotProduct:
    """Tests for dot_product."""

    def test_known_value(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert dot_product(a, b) == pytest.approx(32.0)

    def test_orthogonal(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert dot_product(a, b) == pytest.approx(0.0)

    def test_self_dot_is_norm_squared(self) -> None:
        v = [3.0, 4.0]
        assert dot_product(v, v) == pytest.approx(25.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            dot_product([1.0], [1.0, 2.0])


class TestL2Normalize:
    """Tests for l2_normalize."""

    def test_unit_length(self) -> None:
        v = l2_normalize([3.0, 4.0])
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0)

    def test_zero_vector(self) -> None:
        v = l2_normalize([0.0, 0.0])
        assert v == [0.0, 0.0]

    def test_already_normalised(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert l2_normalize(v) == pytest.approx(v)


class TestComputeScore:
    """Tests for compute_score dispatcher."""

    def test_cosine(self) -> None:
        a = [1.0, 0.0]
        b = [1.0, 0.0]
        assert compute_score(a, b, DistanceMetric.COSINE) == pytest.approx(1.0)

    def test_euclidean_converts_to_similarity(self) -> None:
        a = [0.0, 0.0]
        b = [3.0, 4.0]
        score = compute_score(a, b, DistanceMetric.EUCLIDEAN)
        expected = 1.0 / (1.0 + 5.0)
        assert score == pytest.approx(expected)

    def test_dot_product(self) -> None:
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        assert compute_score(a, b, DistanceMetric.DOT_PRODUCT) == pytest.approx(11.0)


class TestComputePairwiseScores:
    """Tests for compute_pairwise_scores."""

    def test_returns_correct_count(self) -> None:
        query = [1.0, 0.0]
        vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        scores = compute_pairwise_scores(query, vectors, DistanceMetric.COSINE)
        assert len(scores) == 3

    def test_self_is_most_similar(self) -> None:
        query = [1.0, 0.0]
        vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        scores = compute_pairwise_scores(query, vectors, DistanceMetric.COSINE)
        assert scores[0] > scores[1] > scores[2]


# =====================================================================
# VectorStoreConfig tests
# =====================================================================


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig defaults and overrides."""

    def test_defaults(self) -> None:
        config = VectorStoreConfig()
        assert config.collection_name == "default"
        assert config.dimensions == 384
        assert config.metric == DistanceMetric.COSINE

    def test_custom_values(self) -> None:
        config = VectorStoreConfig(
            collection_name="my_index",
            dimensions=768,
            metric=DistanceMetric.DOT_PRODUCT,
            extra={"param": 42},
        )
        assert config.collection_name == "my_index"
        assert config.dimensions == 768
        assert config.metric == DistanceMetric.DOT_PRODUCT
        assert config.extra == {"param": 42}


# =====================================================================
# InMemoryVectorStore tests
# =====================================================================


class TestInMemoryStoreLifecycle:
    """Tests for store initialization and lifecycle."""

    def test_not_initialized_raises(self) -> None:
        store = InMemoryVectorStore()
        with pytest.raises(RuntimeError, match="not initialized"):
            store.add([])

    def test_context_manager(self) -> None:
        config = VectorStoreConfig(dimensions=8)
        with InMemoryVectorStore(config=config) as store:
            assert store.count() == 0
        # After exit, should be de-initialized
        with pytest.raises(RuntimeError, match="not initialized"):
            store.add([])

    def test_double_initialize_resets(self) -> None:
        config = VectorStoreConfig(dimensions=8)
        store = InMemoryVectorStore(config=config)
        store.initialize()
        items = _make_sample_items()
        store.add(items[:2])
        assert store.count() == 2
        store.initialize()  # Reset
        assert store.count() == 0

    def test_count_before_init(self) -> None:
        store = InMemoryVectorStore()
        assert store.count() == 0


class TestInMemoryStoreAdd:
    """Tests for the add operation."""

    def test_add_single(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        ids = memory_store.add([items[0]])
        assert len(ids) == 1
        assert ids[0] == "chunk-0"
        assert memory_store.count() == 1

    def test_add_multiple(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        ids = memory_store.add(items)
        assert len(ids) == 5
        assert memory_store.count() == 5

    def test_add_duplicate_raises(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add([items[0]])
        with pytest.raises(ValueError, match="Duplicate"):
            memory_store.add([items[0]])

    def test_add_wrong_dimensions_raises(self, memory_store: InMemoryVectorStore) -> None:
        item = _make_embedded_chunk(
            content="test",
            vector=[1.0, 2.0, 3.0],  # 3 dims, store expects 8
            chunk_id="bad-dims",
        )
        with pytest.raises(ValueError, match="Expected 8-dim"):
            memory_store.add([item])

    def test_add_one_convenience(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        cid = memory_store.add_one(items[0])
        assert cid == "chunk-0"
        assert memory_store.count() == 1


class TestInMemoryStoreUpsert:
    """Tests for the upsert operation."""

    def test_upsert_new(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        ids = memory_store.upsert([items[0]])
        assert len(ids) == 1
        assert memory_store.count() == 1

    def test_upsert_existing(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add([items[0]])
        # Upsert with same ID but different content
        updated = _make_embedded_chunk(
            content="Updated content",
            vector=MockEmbedder(dimensions=8).embed_text("updated"),
            chunk_id="chunk-0",
        )
        memory_store.upsert([updated])
        assert memory_store.count() == 1
        retrieved = memory_store.get("chunk-0")
        assert retrieved is not None
        assert retrieved.chunk.content == "Updated content"


class TestInMemoryStoreDelete:
    """Tests for the delete operation."""

    def test_delete_existing(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        removed = memory_store.delete(["chunk-0", "chunk-1"])
        assert removed == 2
        assert memory_store.count() == 3

    def test_delete_nonexistent(self, memory_store: InMemoryVectorStore) -> None:
        removed = memory_store.delete(["nonexistent"])
        assert removed == 0

    def test_delete_one_convenience(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items[:1])
        assert memory_store.delete_one("chunk-0") is True
        assert memory_store.count() == 0

    def test_delete_one_nonexistent(self, memory_store: InMemoryVectorStore) -> None:
        assert memory_store.delete_one("ghost") is False

    def test_clear(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        removed = memory_store.clear()
        assert removed == 5
        assert memory_store.count() == 0


class TestInMemoryStoreGet:
    """Tests for the get and list operations."""

    def test_get_existing(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items[:1])
        result = memory_store.get("chunk-0")
        assert result is not None
        assert result.chunk.content == items[0].chunk.content
        assert result.chunk.id == "chunk-0"

    def test_get_nonexistent(self, memory_store: InMemoryVectorStore) -> None:
        assert memory_store.get("ghost") is None

    def test_list_ids(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        ids = memory_store.list_ids()
        assert ids == ["chunk-0", "chunk-1", "chunk-2", "chunk-3", "chunk-4"]

    def test_list_ids_empty(self, memory_store: InMemoryVectorStore) -> None:
        assert memory_store.list_ids() == []

    def test_get_preserves_metadata(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items[:1])
        result = memory_store.get("chunk-0")
        assert result is not None
        assert result.chunk.metadata.get("topic") == "ai"


class TestInMemoryStoreSearch:
    """Tests for similarity search."""

    def test_search_returns_results(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        query = MockEmbedder(dimensions=8).embed_text("machine learning deep learning")
        results = memory_store.search(query, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_ranks_are_sequential(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        query = MockEmbedder(dimensions=8).embed_text("machine learning")
        results = memory_store.search(query, top_k=5)
        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3, 4, 5]

    def test_search_scores_are_descending(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        query = MockEmbedder(dimensions=8).embed_text("neural networks")
        results = memory_store.search(query, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_store(self, memory_store: InMemoryVectorStore) -> None:
        query = [0.0] * 8
        results = memory_store.search(query, top_k=5)
        assert results == []

    def test_search_top_k_limit(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        query = MockEmbedder(dimensions=8).embed_text("test")
        results = memory_store.search(query, top_k=2)
        assert len(results) == 2

    def test_search_wrong_dimensions_raises(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        with pytest.raises(ValueError, match="Expected 8-dim"):
            memory_store.search([1.0, 2.0], top_k=3)

    def test_search_with_document_filter(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        query = MockEmbedder(dimensions=8).embed_text("test")
        # doc-0 has chunks 0, 1, 2; doc-1 has chunks 3, 4
        results = memory_store.search(query, top_k=10, filters={"document_id": "doc-0"})
        assert all(r.chunk.document_id == "doc-0" for r in results)
        assert len(results) == 3

    def test_search_with_metadata_filter(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        query = MockEmbedder(dimensions=8).embed_text("test")
        results = memory_store.search(query, top_k=10, filters={"topic": "ai"})
        assert all(r.chunk.metadata.get("topic") == "ai" for r in results)
        assert len(results) == 3

    def test_search_by_text(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        embedder = MockEmbedder(dimensions=8)
        results = memory_store.search_by_text("neural network", embedder, top_k=3)
        assert len(results) == 3


class TestInMemoryStoreSearchMetrics:
    """Test search with different distance metrics."""

    def test_euclidean_search(self) -> None:
        config = VectorStoreConfig(dimensions=8, metric=DistanceMetric.EUCLIDEAN)
        store = InMemoryVectorStore(config=config)
        store.initialize()
        items = _make_sample_items()
        store.add(items)
        query = MockEmbedder(dimensions=8).embed_text("machine learning")
        results = store.search(query, top_k=3)
        assert len(results) == 3
        # Euclidean scores should be in (0, 1] range
        for r in results:
            assert 0.0 < r.score <= 1.0

    def test_dot_product_search(self) -> None:
        config = VectorStoreConfig(dimensions=8, metric=DistanceMetric.DOT_PRODUCT)
        store = InMemoryVectorStore(config=config)
        store.initialize()
        items = _make_sample_items()
        store.add(items)
        query = MockEmbedder(dimensions=8).embed_text("machine learning")
        results = store.search(query, top_k=3)
        assert len(results) == 3


class TestInMemoryStorePersistence:
    """Tests for JSON persistence."""

    def test_persist_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "store.json")

            # Create and persist
            config = VectorStoreConfig(
                collection_name="persist_test",
                dimensions=8,
                metric=DistanceMetric.COSINE,
            )
            store1 = InMemoryVectorStore(config=config)
            store1.initialize()
            items = _make_sample_items()
            store1.add(items)
            store1.persist(path)

            # Load into new store
            store2 = InMemoryVectorStore()
            store2.load(path)
            assert store2.count() == 5
            assert store2.config.collection_name == "persist_test"
            assert store2.config.metric == DistanceMetric.COSINE

            # Verify data integrity
            for item in items:
                loaded = store2.get(item.chunk.id)
                assert loaded is not None
                assert loaded.chunk.content == item.chunk.content

    def test_persist_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sub" / "dir" / "store.json")
            config = VectorStoreConfig(dimensions=8)
            store = InMemoryVectorStore(config=config)
            store.initialize()
            store.persist(path)
            assert Path(path).exists()

    def test_load_nonexistent_raises(self) -> None:
        store = InMemoryVectorStore()
        with pytest.raises(FileNotFoundError):
            store.load("/nonexistent/path.json")

    def test_persist_empty_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "empty.json")
            config = VectorStoreConfig(dimensions=8)
            store = InMemoryVectorStore(config=config)
            store.initialize()
            store.persist(path)

            # Verify it's valid JSON
            data = json.loads(Path(path).read_text())
            assert data["items"] == {}

    def test_round_trip_preserves_search(self) -> None:
        """Verify search results are identical after persist/load."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "roundtrip.json")
            embedder = MockEmbedder(dimensions=8)
            config = VectorStoreConfig(dimensions=8)

            store1 = InMemoryVectorStore(config=config)
            store1.initialize()
            items = _make_sample_items()
            store1.add(items)
            query = embedder.embed_text("machine learning")
            results1 = store1.search(query, top_k=3)
            store1.persist(path)

            store2 = InMemoryVectorStore()
            store2.load(path)
            results2 = store2.search(query, top_k=3)

            assert len(results1) == len(results2)
            for r1, r2 in zip(results1, results2, strict=False):
                assert r1.chunk.id == r2.chunk.id
                assert r1.score == pytest.approx(r2.score)


class TestInMemoryStoreGetConfig:
    """Tests for get_config."""

    def test_get_config(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        config = memory_store.get_config()
        assert config["collection_name"] == "test_collection"
        assert config["dimensions"] == 8
        assert config["metric"] == "cosine"
        assert config["count"] == 5
        assert config["backend"] == "InMemoryVectorStore"


class TestInMemoryStoreRepr:
    """Tests for __repr__."""

    def test_repr(self, memory_store: InMemoryVectorStore) -> None:
        r = repr(memory_store)
        assert "InMemoryVectorStore" in r
        assert "test_collection" in r
        assert "cosine" in r


class TestInMemoryStoreEdgeCases:
    """Edge case tests."""

    def test_add_empty_list(self, memory_store: InMemoryVectorStore) -> None:
        ids = memory_store.add([])
        assert ids == []
        assert memory_store.count() == 0

    def test_delete_empty_list(self, memory_store: InMemoryVectorStore) -> None:
        removed = memory_store.delete([])
        assert removed == 0

    def test_search_returns_all_when_top_k_exceeds_count(
        self, memory_store: InMemoryVectorStore
    ) -> None:
        items = _make_sample_items()
        memory_store.add(items[:2])
        query = MockEmbedder(dimensions=8).embed_text("test")
        results = memory_store.search(query, top_k=100)
        assert len(results) == 2

    def test_filter_no_matches(self, memory_store: InMemoryVectorStore) -> None:
        items = _make_sample_items()
        memory_store.add(items)
        query = MockEmbedder(dimensions=8).embed_text("test")
        results = memory_store.search(query, top_k=10, filters={"topic": "nonexistent"})
        assert results == []
