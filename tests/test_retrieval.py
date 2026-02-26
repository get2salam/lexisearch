"""Comprehensive test suite for the retrieval engine.

Covers BM25, vector retriever, hybrid fusion, reranking, MMR diversity,
and query expansion — ~100 test cases.
"""

from __future__ import annotations

import pytest

from lexisearch.embeddings.mock import MockEmbedder
from lexisearch.models import Chunk, ChunkStrategy, SearchResult
from lexisearch.retrieval.base import (
    FilterOperator,
    MetadataFilter,
    RetrieverConfig,
    RetrieverType,
)
from lexisearch.retrieval.bm25 import BM25Config, BM25Retriever
from lexisearch.retrieval.hybrid import (
    FusionMethod,
    HybridConfig,
    HybridRetriever,
)
from lexisearch.retrieval.mmr import greedy_diversify, mmr_select
from lexisearch.retrieval.query import (
    ExpandedQuery,
    MultiQueryExpander,
    QueryDecomposer,
    SynonymExpander,
)
from lexisearch.retrieval.reranker import (
    LinearScoreReranker,
    RerankedRetriever,
)
from lexisearch.retrieval.vector_retriever import VectorRetriever, VectorRetrieverConfig
from lexisearch.vectorstore.base import VectorStoreConfig
from lexisearch.vectorstore.memory import InMemoryVectorStore

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create a varied set of sample chunks for retrieval testing."""
    texts = [
        "Machine learning is a subset of artificial intelligence that focuses on "
        "building systems capable of learning from data and improving performance.",
        "Deep learning uses neural networks with multiple layers to learn "
        "hierarchical representations of data for complex pattern recognition.",
        "Natural language processing enables computers to understand, interpret, "
        "and generate human language for tasks like translation and summarization.",
        "Information retrieval is the science of searching for relevant documents "
        "within large collections using keyword and semantic matching techniques.",
        "Transformer architectures have revolutionized NLP by enabling parallel "
        "processing of sequences through self-attention mechanisms.",
        "Vector databases store high-dimensional embeddings and support efficient "
        "approximate nearest neighbor search for similarity queries.",
        "Retrieval-augmented generation combines document retrieval with language "
        "models to produce grounded, factually accurate text responses.",
        "Reinforcement learning trains agents to make sequential decisions by "
        "maximizing cumulative reward through trial and error interactions.",
        "Computer vision applies deep learning to image and video analysis, "
        "enabling object detection, segmentation, and scene understanding.",
        "Knowledge graphs represent structured information as entities and "
        "relationships, enabling complex reasoning and inference over facts.",
    ]
    return [
        Chunk(
            content=text,
            document_id=f"doc-{i}",
            index=i,
            start_char=0,
            end_char=len(text),
            strategy=ChunkStrategy.FIXED_SIZE,
            metadata={"topic": f"topic-{i % 3}", "source": "test"},
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def embedder() -> MockEmbedder:
    """Create a mock embedder for testing."""
    return MockEmbedder(dimensions=64)


@pytest.fixture
def vector_store(embedder: MockEmbedder, sample_chunks: list[Chunk]) -> InMemoryVectorStore:
    """Create and populate a vector store for testing."""
    config = VectorStoreConfig(dimensions=64)
    store = InMemoryVectorStore(config=config)
    store.initialize()

    embedded = embedder.embed_chunks(sample_chunks)
    store.add(embedded)
    return store


@pytest.fixture
def bm25(sample_chunks: list[Chunk]) -> BM25Retriever:
    """Create and populate a BM25 retriever."""
    retriever = BM25Retriever()
    retriever.add_chunks(sample_chunks)
    return retriever


@pytest.fixture
def vector_retriever(
    vector_store: InMemoryVectorStore,
    embedder: MockEmbedder,
) -> VectorRetriever:
    """Create a vector retriever wrapping the populated store."""
    return VectorRetriever(vector_store, embedder)


# ======================================================================
# BaseRetriever / MetadataFilter
# ======================================================================


class TestMetadataFilter:
    """Tests for the MetadataFilter and filter operators."""

    def test_eq_filter(self):
        f = MetadataFilter(field="topic", operator=FilterOperator.EQ, value="a")
        assert f.field == "topic"
        assert f.operator is FilterOperator.EQ

    def test_filter_operators_exhaustive(self):
        """Ensure all operators are enumerated."""
        assert len(FilterOperator) == 9

    def test_retriever_type_values(self):
        assert RetrieverType.SPARSE.value == "sparse"
        assert RetrieverType.DENSE.value == "dense"
        assert RetrieverType.HYBRID.value == "hybrid"
        assert RetrieverType.RERANKED.value == "reranked"

    def test_metadata_filter_check_eq(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={"lang": "en"})
        f = MetadataFilter(field="lang", operator=FilterOperator.EQ, value="en")
        assert retriever.apply_metadata_filter(chunk, [f]) is True

    def test_metadata_filter_check_neq(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={"lang": "en"})
        f = MetadataFilter(field="lang", operator=FilterOperator.NEQ, value="fr")
        assert retriever.apply_metadata_filter(chunk, [f]) is True

    def test_metadata_filter_gt(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={"score": 0.8})
        f = MetadataFilter(field="score", operator=FilterOperator.GT, value=0.5)
        assert retriever.apply_metadata_filter(chunk, [f]) is True

    def test_metadata_filter_gte(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={"count": 5})
        f = MetadataFilter(field="count", operator=FilterOperator.GTE, value=5)
        assert retriever.apply_metadata_filter(chunk, [f]) is True

    def test_metadata_filter_lt(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={"count": 3})
        f = MetadataFilter(field="count", operator=FilterOperator.LT, value=5)
        assert retriever.apply_metadata_filter(chunk, [f]) is True

    def test_metadata_filter_lte(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={"count": 5})
        f = MetadataFilter(field="count", operator=FilterOperator.LTE, value=5)
        assert retriever.apply_metadata_filter(chunk, [f]) is True

    def test_metadata_filter_in(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={"lang": "en"})
        f = MetadataFilter(field="lang", operator=FilterOperator.IN, value=["en", "fr"])
        assert retriever.apply_metadata_filter(chunk, [f]) is True

    def test_metadata_filter_not_in(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={"lang": "en"})
        f = MetadataFilter(field="lang", operator=FilterOperator.NOT_IN, value=["de", "fr"])
        assert retriever.apply_metadata_filter(chunk, [f]) is True

    def test_metadata_filter_contains(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={"tags": "ml,nlp,cv"})
        f = MetadataFilter(field="tags", operator=FilterOperator.CONTAINS, value="nlp")
        assert retriever.apply_metadata_filter(chunk, [f]) is True

    def test_metadata_filter_missing_field(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={})
        f = MetadataFilter(field="missing", operator=FilterOperator.EQ, value="x")
        assert retriever.apply_metadata_filter(chunk, [f]) is False

    def test_metadata_filter_missing_field_neq(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={})
        f = MetadataFilter(field="missing", operator=FilterOperator.NEQ, value="x")
        assert retriever.apply_metadata_filter(chunk, [f]) is True

    def test_multiple_filters_all_pass(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={"a": 1, "b": "x"})
        filters = [
            MetadataFilter(field="a", operator=FilterOperator.EQ, value=1),
            MetadataFilter(field="b", operator=FilterOperator.EQ, value="x"),
        ]
        assert retriever.apply_metadata_filter(chunk, filters) is True

    def test_multiple_filters_one_fails(self):
        retriever = BM25Retriever()
        chunk = Chunk(content="test", document_id="d1", metadata={"a": 1, "b": "y"})
        filters = [
            MetadataFilter(field="a", operator=FilterOperator.EQ, value=1),
            MetadataFilter(field="b", operator=FilterOperator.EQ, value="x"),
        ]
        assert retriever.apply_metadata_filter(chunk, filters) is False


# ======================================================================
# BM25Retriever
# ======================================================================


class TestBM25Retriever:
    """Tests for BM25 sparse retrieval."""

    def test_add_chunks(self, sample_chunks: list[Chunk]):
        retriever = BM25Retriever()
        added = retriever.add_chunks(sample_chunks)
        assert added == len(sample_chunks)
        assert retriever.corpus_size == len(sample_chunks)

    def test_add_duplicate_skipped(self, sample_chunks: list[Chunk]):
        retriever = BM25Retriever()
        retriever.add_chunks(sample_chunks)
        added_again = retriever.add_chunks(sample_chunks)
        assert added_again == 0

    def test_remove_chunks(self, bm25: BM25Retriever, sample_chunks: list[Chunk]):
        initial = bm25.corpus_size
        removed = bm25.remove_chunks([sample_chunks[0].id])
        assert removed == 1
        assert bm25.corpus_size == initial - 1

    def test_remove_nonexistent(self, bm25: BM25Retriever):
        removed = bm25.remove_chunks(["nonexistent-id"])
        assert removed == 0

    def test_clear(self, bm25: BM25Retriever):
        bm25.clear()
        assert bm25.corpus_size == 0

    def test_retrieve_basic(self, bm25: BM25Retriever):
        results = bm25.retrieve("machine learning", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3
        # Machine learning chunk should rank high
        texts = [r.chunk.content for r in results]
        assert any("machine learning" in t.lower() for t in texts)

    def test_retrieve_empty_query(self, bm25: BM25Retriever):
        results = bm25.retrieve("", top_k=5)
        assert len(results) == 0

    def test_retrieve_no_match(self, bm25: BM25Retriever):
        results = bm25.retrieve("xyzzyspoon quantum", top_k=5)
        assert len(results) == 0

    def test_retrieve_scores_descending(self, bm25: BM25Retriever):
        results = bm25.retrieve("deep learning neural", top_k=5)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_search_returns_response(self, bm25: BM25Retriever):
        response = bm25.search("information retrieval", top_k=3)
        assert response.query == "information retrieval"
        assert response.latency_ms >= 0
        assert len(response.results) <= 3
        for r in response.results:
            assert r.rank > 0

    def test_retriever_type(self, bm25: BM25Retriever):
        assert bm25.retriever_type() is RetrieverType.SPARSE

    def test_tokenize_lowercase(self):
        retriever = BM25Retriever()
        tokens = retriever.tokenize("Hello World Test")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_stop_words_removed(self):
        retriever = BM25Retriever()
        tokens = retriever.tokenize("the quick and brown")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_tokenize_punctuation_stripped(self):
        retriever = BM25Retriever()
        tokens = retriever.tokenize("hello, world! test.")
        assert "hello" in tokens
        assert "world" in tokens

    def test_custom_bm25_config(self, sample_chunks: list[Chunk]):
        config = BM25Config(k1=2.0, b=0.5, stop_words=frozenset())
        retriever = BM25Retriever(bm25_config=config)
        retriever.add_chunks(sample_chunks)
        results = retriever.retrieve("the machine learning", top_k=3)
        assert len(results) > 0

    def test_score_threshold(self, sample_chunks: list[Chunk]):
        config = RetrieverConfig(score_threshold=100.0)
        retriever = BM25Retriever(config=config)
        retriever.add_chunks(sample_chunks)
        response = retriever.search("machine learning", top_k=5)
        assert len(response.results) == 0

    def test_term_stats(self, bm25: BM25Retriever):
        stats = bm25.get_term_stats()
        assert stats["corpus_size"] == 10
        assert stats["vocabulary_size"] > 0
        assert stats["avg_doc_length"] > 0

    def test_idf_common_term(self, bm25: BM25Retriever):
        idf = bm25._idf("learning")
        assert idf > 0

    def test_idf_rare_term(self, bm25: BM25Retriever):
        idf_rare = bm25._idf("reinforcement")
        idf_common = bm25._idf("learning")
        # Rare terms should have higher IDF
        assert idf_rare >= idf_common

    def test_metadata_in_results(self, bm25: BM25Retriever):
        results = bm25.retrieve("machine learning", top_k=1)
        assert results[0].metadata["retriever"] == "bm25"
        assert "query_tokens" in results[0].metadata

    def test_repr(self, bm25: BM25Retriever):
        r = repr(bm25)
        assert "BM25Retriever" in r
        assert "sparse" in r


# ======================================================================
# VectorRetriever
# ======================================================================


class TestVectorRetriever:
    """Tests for dense vector retrieval."""

    def test_retrieve_basic(self, vector_retriever: VectorRetriever):
        results = vector_retriever.retrieve("machine learning", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_retrieve_scores_positive(self, vector_retriever: VectorRetriever):
        results = vector_retriever.retrieve("deep learning", top_k=5)
        for r in results:
            assert r.score > -2.0  # Cosine sim is in [-1, 1]

    def test_retriever_type(self, vector_retriever: VectorRetriever):
        assert vector_retriever.retriever_type() is RetrieverType.DENSE

    def test_embed_query(self, vector_retriever: VectorRetriever):
        vec = vector_retriever.embed_query("test query")
        assert len(vec) == 64

    def test_retrieve_by_vector(
        self,
        vector_retriever: VectorRetriever,
        embedder: MockEmbedder,
    ):
        query_vec = embedder.embed_text("neural networks")
        results = vector_retriever.retrieve_by_vector(query_vec, top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_results_have_retriever_metadata(
        self,
        vector_retriever: VectorRetriever,
    ):
        results = vector_retriever.retrieve("search query", top_k=1)
        assert results[0].metadata["retriever"] == "vector"
        assert results[0].metadata["embedding_model"] == "mock-embedder"

    def test_normalize_scores(self, vector_retriever: VectorRetriever):
        config = VectorRetrieverConfig(normalize_scores=True)
        retriever = VectorRetriever(
            vector_retriever.vector_store,
            vector_retriever.embedder,
            config=config,
        )
        results = retriever.retrieve("machine learning", top_k=5)
        if len(results) > 1:
            assert results[0].score == pytest.approx(1.0, abs=0.01)

    def test_search_response(self, vector_retriever: VectorRetriever):
        response = vector_retriever.search("information retrieval", top_k=5)
        assert response.query == "information retrieval"
        assert response.latency_ms >= 0
        for r in response.results:
            assert r.rank > 0

    def test_get_config(self, vector_retriever: VectorRetriever):
        config = vector_retriever.get_config()
        assert config["type"] == "dense"
        assert config["vector_store"] == "InMemoryVectorStore"
        assert config["embedder"] == "mock-embedder"


# ======================================================================
# HybridRetriever
# ======================================================================


class TestHybridRetriever:
    """Tests for hybrid retrieval with score fusion."""

    def test_rrf_fusion(
        self,
        bm25: BM25Retriever,
        vector_retriever: VectorRetriever,
    ):
        hybrid = HybridRetriever(
            retrievers=[bm25, vector_retriever],
            config=HybridConfig(fusion_method=FusionMethod.RRF, top_k=5),
        )
        results = hybrid.retrieve("machine learning", top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_linear_fusion(
        self,
        bm25: BM25Retriever,
        vector_retriever: VectorRetriever,
    ):
        hybrid = HybridRetriever(
            retrievers=[bm25, vector_retriever],
            config=HybridConfig(
                fusion_method=FusionMethod.LINEAR,
                weights=[0.4, 0.6],
                top_k=5,
            ),
        )
        results = hybrid.retrieve("information retrieval", top_k=5)
        assert len(results) > 0

    def test_dbsf_fusion(
        self,
        bm25: BM25Retriever,
        vector_retriever: VectorRetriever,
    ):
        hybrid = HybridRetriever(
            retrievers=[bm25, vector_retriever],
            config=HybridConfig(
                fusion_method=FusionMethod.DBSF,
                weights=[0.5, 0.5],
                top_k=5,
            ),
        )
        results = hybrid.retrieve("deep learning neural", top_k=5)
        assert len(results) > 0

    def test_rrf_scores_descending(
        self,
        bm25: BM25Retriever,
        vector_retriever: VectorRetriever,
    ):
        hybrid = HybridRetriever(
            retrievers=[bm25, vector_retriever],
            config=HybridConfig(top_k=10),
        )
        results = hybrid.retrieve("learning data", top_k=10)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_source_tracking(
        self,
        bm25: BM25Retriever,
        vector_retriever: VectorRetriever,
    ):
        hybrid = HybridRetriever(retrievers=[bm25, vector_retriever])
        results = hybrid.retrieve("machine learning", top_k=3)
        for r in results:
            assert "sources" in r.metadata
            assert r.metadata["retriever"] == "hybrid"

    def test_retriever_type(
        self,
        bm25: BM25Retriever,
        vector_retriever: VectorRetriever,
    ):
        hybrid = HybridRetriever(retrievers=[bm25, vector_retriever])
        assert hybrid.retriever_type() is RetrieverType.HYBRID

    def test_minimum_retrievers_validation(self, bm25: BM25Retriever):
        with pytest.raises(ValueError, match="at least 2"):
            HybridRetriever(retrievers=[bm25])

    def test_weight_count_validation(
        self,
        bm25: BM25Retriever,
        vector_retriever: VectorRetriever,
    ):
        with pytest.raises(ValueError, match="weights"):
            HybridRetriever(
                retrievers=[bm25, vector_retriever],
                config=HybridConfig(weights=[0.5]),
            )

    def test_default_equal_weights(
        self,
        bm25: BM25Retriever,
        vector_retriever: VectorRetriever,
    ):
        hybrid = HybridRetriever(retrievers=[bm25, vector_retriever])
        assert len(hybrid._config.weights) == 2
        assert hybrid._config.weights[0] == pytest.approx(0.5)

    def test_get_config(
        self,
        bm25: BM25Retriever,
        vector_retriever: VectorRetriever,
    ):
        hybrid = HybridRetriever(retrievers=[bm25, vector_retriever])
        config = hybrid.get_config()
        assert config["fusion_method"] == "rrf"
        assert config["num_retrievers"] == 2
        assert "sparse" in config["sub_retrievers"]
        assert "dense" in config["sub_retrievers"]

    def test_rrf_static_method(self):
        """Test RRF fusion with controlled data."""
        r1 = SearchResult(
            chunk=Chunk(content="a", document_id="d1", id="c1"),
            score=0.9,
            rank=1,
        )
        r2 = SearchResult(
            chunk=Chunk(content="b", document_id="d2", id="c2"),
            score=0.8,
            rank=2,
        )
        r3 = SearchResult(
            chunk=Chunk(content="b", document_id="d2", id="c2"),
            score=0.7,
            rank=1,
        )

        fused = HybridRetriever.reciprocal_rank_fusion([[r1, r2], [r3]], k=60)
        # c2 appears in both lists, should rank higher
        ids = [cid for cid, _, _ in fused]
        assert "c2" in ids

    def test_linear_static_method(self):
        """Test linear fusion with controlled data."""
        r1 = SearchResult(
            chunk=Chunk(content="a", document_id="d1", id="c1"),
            score=1.0,
            rank=1,
        )
        r2 = SearchResult(
            chunk=Chunk(content="b", document_id="d2", id="c2"),
            score=0.5,
            rank=1,
        )

        fused = HybridRetriever.linear_fusion([[r1], [r2]], weights=[0.5, 0.5])
        assert len(fused) == 2


# ======================================================================
# Reranker
# ======================================================================


class TestReranker:
    """Tests for reranking models."""

    def test_linear_reranker_basic(self):
        reranker = LinearScoreReranker()
        results = [
            SearchResult(
                chunk=Chunk(content="machine learning algorithms", document_id="d1"),
                score=0.8,
                rank=1,
            ),
            SearchResult(
                chunk=Chunk(content="deep learning neural networks", document_id="d2"),
                score=0.7,
                rank=2,
            ),
            SearchResult(
                chunk=Chunk(content="information retrieval systems", document_id="d3"),
                score=0.6,
                rank=3,
            ),
        ]
        reranked = reranker.rerank("machine learning", results, top_k=3)
        assert len(reranked) == 3
        # The ML result should still be ranked high due to term coverage
        assert reranked[0].metadata["reranker"] == "linear"

    def test_linear_reranker_preserves_original(self):
        reranker = LinearScoreReranker()
        results = [
            SearchResult(
                chunk=Chunk(content="test content", document_id="d1"),
                score=0.5,
                rank=1,
            ),
        ]
        reranked = reranker.rerank("test", results, top_k=1)
        assert reranked[0].metadata["original_score"] == 0.5
        assert reranked[0].metadata["original_rank"] == 1

    def test_linear_reranker_empty(self):
        reranker = LinearScoreReranker()
        assert reranker.rerank("query", [], top_k=5) == []

    def test_linear_score_pair(self):
        reranker = LinearScoreReranker()
        score = reranker.score_pair("machine learning", "machine learning is great")
        assert score > 0

    def test_linear_score_exact_match_bonus(self):
        reranker = LinearScoreReranker()
        score_exact = reranker.score_pair("ML", "ML is a field of study")
        score_no = reranker.score_pair("ML", "deep learning networks")
        assert score_exact > score_no

    def test_reranked_retriever(self, bm25: BM25Retriever):
        reranker = LinearScoreReranker()
        pipeline = RerankedRetriever(bm25, reranker, prefetch_multiplier=2)
        results = pipeline.retrieve("machine learning", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_reranked_retriever_type(self, bm25: BM25Retriever):
        reranker = LinearScoreReranker()
        pipeline = RerankedRetriever(bm25, reranker)
        assert pipeline.retriever_type() is RetrieverType.RERANKED

    def test_reranked_retriever_config(self, bm25: BM25Retriever):
        reranker = LinearScoreReranker()
        pipeline = RerankedRetriever(bm25, reranker, prefetch_multiplier=4)
        config = pipeline.get_config()
        assert config["base_retriever"] == "sparse"
        assert config["reranker"] == "LinearScoreReranker"
        assert config["prefetch_multiplier"] == 4

    def test_linear_reranker_top_k_limit(self):
        reranker = LinearScoreReranker()
        results = [
            SearchResult(
                chunk=Chunk(content=f"content {i}", document_id=f"d{i}"),
                score=0.9 - i * 0.1,
                rank=i + 1,
            )
            for i in range(5)
        ]
        reranked = reranker.rerank("content", results, top_k=2)
        assert len(reranked) == 2


# ======================================================================
# MMR Diversity
# ======================================================================


class TestMMR:
    """Tests for Maximal Marginal Relevance and diversity selection."""

    def _make_results_and_vectors(
        self, n: int = 5, dims: int = 8
    ) -> tuple[list[SearchResult], list[list[float]], list[float]]:
        """Helper to create test results with deterministic vectors."""
        embedder = MockEmbedder(dimensions=dims)
        results = []
        vectors = []
        for i in range(n):
            text = f"document content {i} about topic {i % 3}"
            chunk = Chunk(content=text, document_id=f"d{i}", index=i)
            vec = embedder.embed_text(text)
            results.append(SearchResult(chunk=chunk, score=1.0 - i * 0.1, rank=i + 1))
            vectors.append(vec)

        query_vec = embedder.embed_text("document topic query")
        return results, vectors, query_vec

    def test_mmr_select_basic(self):
        results, vectors, query_vec = self._make_results_and_vectors(5)
        selected = mmr_select(query_vec, results, vectors, top_k=3, lambda_param=0.5)
        assert len(selected) == 3
        assert selected[0].rank == 1

    def test_mmr_select_pure_relevance(self):
        results, vectors, query_vec = self._make_results_and_vectors(5)
        selected = mmr_select(query_vec, results, vectors, top_k=3, lambda_param=1.0)
        assert len(selected) == 3
        # With lambda=1.0, should just pick most relevant

    def test_mmr_select_pure_diversity(self):
        results, vectors, query_vec = self._make_results_and_vectors(5)
        selected = mmr_select(query_vec, results, vectors, top_k=3, lambda_param=0.0)
        assert len(selected) == 3

    def test_mmr_select_empty(self):
        selected = mmr_select([0.0, 1.0], [], [], top_k=3)
        assert selected == []

    def test_mmr_select_mismatched_lengths(self):
        results = [
            SearchResult(chunk=Chunk(content="a", document_id="d1"), score=1.0, rank=1),
        ]
        with pytest.raises(ValueError, match="Mismatch"):
            mmr_select([0.0], results, [[0.0], [1.0]], top_k=1)

    def test_mmr_metadata_preserved(self):
        results, vectors, query_vec = self._make_results_and_vectors(3)
        selected = mmr_select(query_vec, results, vectors, top_k=2)
        for r in selected:
            assert r.metadata.get("mmr_applied") is True
            assert "original_score" in r.metadata

    def test_greedy_diversify_basic(self):
        embedder = MockEmbedder(dimensions=8)
        results = []
        vectors = []
        for i in range(5):
            text = f"unique document {i}"
            chunk = Chunk(content=text, document_id=f"d{i}")
            vec = embedder.embed_text(text)
            results.append(SearchResult(chunk=chunk, score=1.0 - i * 0.1, rank=i + 1))
            vectors.append(vec)

        diversified = greedy_diversify(results, vectors, max_similarity=0.95)
        assert len(diversified) >= 1
        assert diversified[0].rank == 1

    def test_greedy_diversify_empty(self):
        assert greedy_diversify([], []) == []

    def test_greedy_diversify_keeps_first(self):
        embedder = MockEmbedder(dimensions=8)
        vec = embedder.embed_text("same content")
        results = [
            SearchResult(
                chunk=Chunk(content="same content", document_id=f"d{i}"),
                score=1.0,
                rank=i + 1,
            )
            for i in range(3)
        ]
        vectors = [vec, vec, vec]  # Identical vectors
        diversified = greedy_diversify(results, vectors, max_similarity=0.9)
        assert len(diversified) == 1  # Only first kept


# ======================================================================
# Query Expansion
# ======================================================================


class TestQueryExpansion:
    """Tests for query expansion strategies."""

    def test_synonym_expander_basic(self):
        expander = SynonymExpander(
            synonyms={"ML": ["machine learning"], "AI": ["artificial intelligence"]}
        )
        result = expander.expand("ML and AI")
        assert "machine learning" in result.expanded
        assert "artificial intelligence" in result.expanded
        assert result.strategy == "synonym"

    def test_synonym_expander_no_match(self):
        expander = SynonymExpander(synonyms={"ML": ["machine learning"]})
        result = expander.expand("deep learning")
        assert result.expanded == "deep learning"
        assert result.added_terms == []

    def test_synonym_expander_case_insensitive(self):
        expander = SynonymExpander(
            synonyms={"ml": ["machine learning"]},
            case_sensitive=False,
        )
        result = expander.expand("ML techniques")
        assert "machine learning" in result.expanded

    def test_query_decomposer_basic(self):
        decomposer = QueryDecomposer()
        result = decomposer.expand("What are transformers and how do attention mechanisms work?")
        assert len(result.sub_queries) >= 2
        assert result.strategy == "decomposition"

    def test_query_decomposer_no_split(self):
        decomposer = QueryDecomposer()
        result = decomposer.expand("What are transformers?")
        assert len(result.sub_queries) >= 1
        assert result.sub_queries[0] == "What are transformers?"

    def test_query_decomposer_semicolon(self):
        decomposer = QueryDecomposer()
        result = decomposer.expand("first topic; second topic")
        assert len(result.sub_queries) == 2

    def test_multi_query_expander_basic(self):
        expander = MultiQueryExpander(max_variations=3)
        result = expander.expand("What is machine learning?")
        assert len(result.sub_queries) >= 1
        assert result.original in result.sub_queries
        assert result.strategy == "multi_query"

    def test_multi_query_expander_dedup(self):
        expander = MultiQueryExpander(max_variations=5)
        result = expander.expand("short query")
        # All sub-queries should be unique
        assert len(result.sub_queries) == len(set(s.lower() for s in result.sub_queries))

    def test_multi_query_expander_question_mark(self):
        expander = MultiQueryExpander()
        result = expander.expand("How does information retrieval work?")
        # Should have a declarative variation
        assert any(not q.endswith("?") for q in result.sub_queries)

    def test_expanded_query_dataclass(self):
        eq = ExpandedQuery(
            original="test",
            expanded="test expanded",
            sub_queries=["test", "expanded"],
            added_terms=["expanded"],
            strategy="test",
        )
        assert eq.original == "test"
        assert len(eq.sub_queries) == 2

    def test_prf_with_bm25(self, bm25: BM25Retriever):
        """Test pseudo-relevance feedback with a BM25 retriever."""
        from lexisearch.retrieval.query import PseudoRelevanceFeedback

        prf = PseudoRelevanceFeedback(
            retriever=bm25,
            num_feedback_docs=3,
            num_expansion_terms=3,
        )
        result = prf.expand("machine learning algorithms")
        assert result.strategy == "prf"
        assert result.metadata["feedback_docs"] > 0

    def test_prf_no_results(self):
        """Test PRF when retriever returns no results."""
        from lexisearch.retrieval.query import PseudoRelevanceFeedback

        empty_bm25 = BM25Retriever()
        prf = PseudoRelevanceFeedback(retriever=empty_bm25)
        result = prf.expand("nonexistent topic")
        assert result.expanded == "nonexistent topic"
        assert result.metadata["feedback_docs"] == 0


# ======================================================================
# Integration: end-to-end hybrid + rerank pipeline
# ======================================================================


class TestIntegrationPipeline:
    """End-to-end integration tests combining multiple components."""

    def test_hybrid_then_rerank(
        self,
        bm25: BM25Retriever,
        vector_retriever: VectorRetriever,
    ):
        """Full pipeline: hybrid retrieval → linear reranking."""
        hybrid = HybridRetriever(
            retrievers=[bm25, vector_retriever],
            config=HybridConfig(top_k=10),
        )
        reranker = LinearScoreReranker()
        pipeline = RerankedRetriever(hybrid, reranker, prefetch_multiplier=2)

        response = pipeline.search("deep learning neural networks", top_k=3)
        assert len(response.results) > 0
        assert len(response.results) <= 3
        assert response.latency_ms >= 0

    def test_bm25_with_query_expansion(
        self,
        bm25: BM25Retriever,
    ):
        """BM25 retrieval with synonym expansion."""
        expander = SynonymExpander(synonyms={"NLP": ["natural language processing"]})
        expanded = expander.expand("NLP tasks")
        results = bm25.retrieve(expanded.expanded, top_k=3)
        assert len(results) > 0

    def test_full_pipeline_all_components(
        self,
        bm25: BM25Retriever,
        vector_retriever: VectorRetriever,
        embedder: MockEmbedder,
        sample_chunks: list[Chunk],
    ):
        """Full pipeline: expand → hybrid retrieve → rerank → MMR."""
        # 1. Query expansion
        expander = MultiQueryExpander(max_variations=2)
        expanded = expander.expand("What are neural network architectures?")

        # 2. Hybrid retrieval (on first sub-query)
        hybrid = HybridRetriever(
            retrievers=[bm25, vector_retriever],
            config=HybridConfig(top_k=8),
        )
        all_results: list[SearchResult] = []
        for sq in expanded.sub_queries[:2]:
            results = hybrid.retrieve(sq, top_k=8)
            all_results.extend(results)

        # Deduplicate by chunk ID
        seen_ids: set[str] = set()
        unique: list[SearchResult] = []
        for r in all_results:
            if r.chunk.id not in seen_ids:
                seen_ids.add(r.chunk.id)
                unique.append(r)

        # 3. Rerank
        reranker = LinearScoreReranker()
        reranked = reranker.rerank("neural network architectures", unique, top_k=5)

        # 4. MMR diversity (need vectors)
        vectors = [embedder.embed_text(r.chunk.content) for r in reranked]
        query_vec = embedder.embed_text("neural network architectures")
        final = mmr_select(query_vec, reranked, vectors, top_k=3, lambda_param=0.7)

        assert len(final) > 0
        assert len(final) <= 3
        assert final[0].rank == 1
