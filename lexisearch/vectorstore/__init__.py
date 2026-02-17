"""Vector store layer for LexiSearch.

Provides abstract and concrete vector store implementations for
indexing and retrieving embedded document chunks.

Supported backends:
    - :class:`InMemoryVectorStore` — Pure-Python brute-force store (testing/dev).
    - :class:`FAISSVectorStore` — Facebook AI Similarity Search (production).
    - :class:`ChromaVectorStore` — ChromaDB persistent store.
    - :class:`QdrantVectorStore` — Qdrant vector database client.
"""

from __future__ import annotations

from lexisearch.vectorstore.base import BaseVectorStore, DistanceMetric

__all__ = [
    "BaseVectorStore",
    "DistanceMetric",
]
