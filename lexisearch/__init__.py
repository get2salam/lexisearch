"""LexiSearch — A production-ready RAG framework for intelligent document search.

LexiSearch provides a modular pipeline for ingesting documents, chunking text,
generating embeddings, and retrieving relevant content for LLM-powered
applications.

Example:
    >>> from lexisearch import Document, DocumentMetadata
    >>> doc = Document(content="Hello, world!")
    >>> doc.word_count
    2
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Chunk",
    "ChunkStrategy",
    "Document",
    "DocumentFormat",
    "DocumentMetadata",
    "EmbeddedChunk",
    "Embedding",
    "SearchResponse",
    "SearchResult",
]
