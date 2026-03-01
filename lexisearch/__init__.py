"""LexiSearch — A production-ready RAG framework for intelligent document search.

LexiSearch provides a modular pipeline for ingesting documents, chunking text,
generating embeddings, storing vectors, and retrieving relevant content for
LLM-powered applications.

Example:
    >>> from lexisearch import Document, DocumentMetadata
    >>> doc = Document(content="Hello, world!")
    >>> doc.word_count
    2

Pipeline quick start:

    >>> from lexisearch.pipeline import PipelineBuilder, PipelineRunner
    >>> from lexisearch.embeddings import MockEmbedder
    >>> from lexisearch.generation import MockLLM
    >>> from lexisearch.models import Document
    >>> pipeline = (
    ...     PipelineBuilder.create("demo")
    ...     .embed(MockEmbedder())
    ...     .store()
    ...     .retrieve()
    ...     .generate(MockLLM())
    ...     .build()
    ... )
    >>> runner = PipelineRunner(pipeline)
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

__version__ = "0.6.0"

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
