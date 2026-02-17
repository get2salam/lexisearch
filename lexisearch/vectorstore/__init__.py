"""Vector store layer for LexiSearch.

Provides abstract and concrete vector store implementations for
indexing and retrieving embedded document chunks.

Supported backends:
    - :class:`InMemoryVectorStore` — Pure-Python brute-force store (testing/dev).
    - :class:`FAISSVectorStore` — Facebook AI Similarity Search (production).
    - :class:`ChromaVectorStore` — ChromaDB persistent store.
    - :class:`QdrantVectorStore` — Qdrant vector database client.

Quick start::

    from lexisearch.vectorstore import InMemoryVectorStore, VectorStoreConfig

    config = VectorStoreConfig(dimensions=384)
    with InMemoryVectorStore(config=config) as store:
        store.add(embedded_chunks)
        results = store.search(query_vector, top_k=5)
"""

from __future__ import annotations

from lexisearch.vectorstore.base import BaseVectorStore, DistanceMetric, VectorStoreConfig
from lexisearch.vectorstore.memory import InMemoryVectorStore
from lexisearch.vectorstore.metrics import (
    compute_pairwise_scores,
    compute_score,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    l2_normalize,
)

__all__ = [
    "BaseVectorStore",
    "DistanceMetric",
    "InMemoryVectorStore",
    "VectorStoreConfig",
    "compute_pairwise_scores",
    "compute_score",
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
    "l2_normalize",
]
