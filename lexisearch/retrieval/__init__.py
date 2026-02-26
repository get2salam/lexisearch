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
from lexisearch.retrieval.vector_retriever import VectorRetriever, VectorRetrieverConfig

__all__ = [
    # Retrievers
    "BM25Config",
    "BM25Retriever",
    # Query expansion
    "BaseQueryExpander",
    # Rerankers
    "BaseReranker",
    # Base
    "BaseRetriever",
    "CohereReranker",
    "CrossEncoderReranker",
    "ExpandedQuery",
    "FilterOperator",
    "FusionMethod",
    "HybridConfig",
    "HybridRetriever",
    "LinearScoreReranker",
    "MetadataFilter",
    "MultiQueryExpander",
    "PseudoRelevanceFeedback",
    "QueryDecomposer",
    "RerankedRetriever",
    "RerankerConfig",
    "RetrieverConfig",
    "RetrieverType",
    "SynonymExpander",
    "VectorRetriever",
    "VectorRetrieverConfig",
    # Diversity
    "greedy_diversify",
    "mmr_select",
]
