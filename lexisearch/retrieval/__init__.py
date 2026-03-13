"""Retrieval engine for LexiSearch.

Provides sparse, dense, and hybrid retrieval strategies with reranking,
diversity selection, and query expansion utilities.

Retriever hierarchy::

    BaseRetriever
    ├── BM25Retriever          (sparse / keyword)
    ├── VectorRetriever        (dense / embedding)
    ├── HybridRetriever        (sparse + dense fusion)
    └── RerankedRetriever      (two-stage: retrieve → rerank)

Quick start::

    from lexisearch.retrieval import BM25Retriever

    retriever = BM25Retriever()
    retriever.add_chunks(chunks)
    response = retriever.search("information retrieval", top_k=5)
    for result in response.results:
        print(f"[{result.score:.3f}] {result.chunk.content[:80]}")
"""

from __future__ import annotations

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
from lexisearch.retrieval.base import (
    BaseRetriever,
    FilterOperator,
    MetadataFilter,
    RetrieverConfig,
    RetrieverType,
)
from lexisearch.retrieval.bm25 import BM25Config, BM25Retriever
from lexisearch.retrieval.hybrid import FusionMethod, HybridConfig, HybridRetriever
from lexisearch.retrieval.mmr import greedy_diversify, mmr_select
from lexisearch.retrieval.query import (
    BaseQueryExpander,
    ExpandedQuery,
    MultiQueryExpander,
    PseudoRelevanceFeedback,
    QueryDecomposer,
    SynonymExpander,
)
from lexisearch.retrieval.reranker import (
    BaseReranker,
    CohereReranker,
    CrossEncoderReranker,
    LinearScoreReranker,
    RerankedRetriever,
    RerankerConfig,
)
from lexisearch.retrieval.compression import (
    BaseCompressor,
    CompressedChunk,
    KeywordCompressor,
    SentenceCompressor,
)
from lexisearch.retrieval.router import (
    IntentClassification,
    IntentClassifier,
    QueryIntent,
    QueryRouter,
    RoutingResult,
)
from lexisearch.retrieval.vector_retriever import VectorRetriever, VectorRetrieverConfig

__all__ = [
    "AdvancedRetrievalConfig",
    "AdvancedRetrievalResult",
    "BM25Config",
    "BM25Retriever",
    "BaseAdvancedRetriever",
    "BaseQueryExpander",
    "BaseReranker",
    "BaseRetriever",
    "CohereReranker",
    "CompositeAdvancedRetriever",
    "CrossEncoderReranker",
    "ExpandedQuery",
    "FilterOperator",
    "FusionMethod",
    "HyDERetriever",
    "HybridConfig",
    "HybridRetriever",
    "LinearScoreReranker",
    "MetadataFilter",
    "MultiQueryExpander",
    "MultiQueryRetriever",
    "PseudoRelevanceFeedback",
    "QueryDecomposer",
    "RerankedRetriever",
    "RerankerConfig",
    "RetrievedChunk",
    "RetrieverConfig",
    "RetrieverType",
    "RuleBasedQueryGenerator",
    "StepBackRetriever",
    "SynonymExpander",
    "VectorRetriever",
    "VectorRetrieverConfig",
    "BaseCompressor",
    "CompressedChunk",
    "IntentClassification",
    "IntentClassifier",
    "KeywordCompressor",
    "QueryIntent",
    "QueryRouter",
    "RoutingResult",
    "SentenceCompressor",
    "greedy_diversify",
    "mmr_select",
    "reciprocal_rank_fusion",
]
