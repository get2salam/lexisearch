"""LexiSearch — A production-ready RAG framework for intelligent document search.

LexiSearch provides a modular pipeline for ingesting documents, chunking text,
generating embeddings, storing vectors, and retrieving relevant content for
LLM-powered applications.

Example:
    >>> from lexisearch import Document, DocumentMetadata
    >>> doc = Document(content="Hello, world!")
    >>> doc.word_count
    2

Vector store quick start:

    >>> from lexisearch.vectorstore import InMemoryVectorStore, VectorStoreConfig
    >>> config = VectorStoreConfig(dimensions=384)
    >>> store = InMemoryVectorStore(config=config)
"""

from __future__ import annotations

from lexisearch.models import (
    Chunk,
    ChunkStrategy,
    Document,
    DocumentFormat,
    DocumentMetadata,
    EmbeddedChunk,
    Embedding,
    SearchResponse,
    SearchResult,
)

__version__ = "0.3.0"

__all__ = [
    "Chunk",
    "ChunkStrategy",
    "Document",
    "DocumentFormat",
    "DocumentMetadata",
    "EmbeddedChunk",
    "Embedding",
    "SearchResponse",
    "SearchResult",
    "__version__",
]
