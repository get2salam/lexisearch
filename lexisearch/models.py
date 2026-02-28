"""Core data models for LexiSearch.

This module defines the fundamental data structures used throughout the
LexiSearch pipeline: documents, chunks, embeddings, and search results.
All models are implemented as dataclasses with full type annotations.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DocumentFormat(Enum):
    """Supported document formats."""

    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class ChunkStrategy(Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata associated with a document.

    Attributes:
        source: Origin path, URL, or identifier for the document.
        title: Human-readable title of the document.
        author: Author or creator of the document.
        created_at: Timestamp when the document was created or ingested.
        format: The format of the source document.
        language: ISO 639-1 language code (e.g., ``"en"``).
        page_count: Number of pages (for paginated formats like PDF).
        extra: Arbitrary additional metadata as key-value pairs.
    """

    source: str = ""
    title: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    format: DocumentFormat = DocumentFormat.UNKNOWN
    language: str = "en"
    page_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A document ingested into the LexiSearch pipeline.

    Documents are the top-level unit of content. Each document has a unique
    identifier, textual content, and associated metadata.

    Attributes:
        id: Unique identifier for the document.
        content: The full text content of the document.
        metadata: Structured metadata about the document.

    Examples:
        >>> doc = Document(content="Hello, world!")
        >>> len(doc.content)
        13
        >>> doc.content_hash[:8]
        '315f5bdb'
    """

    content: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def content_hash(self) -> str:
        """Compute a SHA-256 hash of the document content.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def char_count(self) -> int:
        """Return the number of characters in the document.

        Returns:
            Character count of the content.
        """
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Return the approximate word count of the document.

        Returns:
            Word count based on whitespace splitting.
        """
        return len(self.content.split())

    def __repr__(self) -> str:
        """Return a concise string representation."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.id!r}, content={preview!r})"


@dataclass
class Chunk:
    """A chunk of text derived from a document.

    Chunks are produced by the chunking pipeline and represent a segment
    of a larger document suitable for embedding and retrieval.

    Attributes:
        id: Unique identifier for the chunk.
        content: The text content of this chunk.
        document_id: Identifier of the source document.
        index: Zero-based position of this chunk within the document.
        start_char: Starting character offset in the original document.
        end_char: Ending character offset in the original document.
        metadata: Additional metadata inherited or computed.
        strategy: The chunking strategy that produced this chunk.
    """

    content: str
    document_id: str
    index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def content_hash(self) -> str:
        """Compute a SHA-256 hash of the chunk content.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def char_count(self) -> int:
        """Return the number of characters in the chunk.

        Returns:
            Character count of the content.
        """
        return len(self.content)

    @property
    def token_estimate(self) -> int:
        """Estimate the token count (rough: chars / 4).

        Returns:
            Approximate token count.
        """
        return max(1, len(self.content) // 4)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        preview = self.content[:40] + "..." if len(self.content) > 40 else self.content
        return (
            f"Chunk(id={self.id!r}, doc={self.document_id!r}, "
            f"index={self.index}, content={preview!r})"
        )


@dataclass
class Embedding:
    """A vector embedding for a chunk of text.

    Attributes:
        chunk_id: Identifier of the chunk this embedding represents.
        vector: The embedding vector as a list of floats.
        model: Name of the embedding model used.
        dimensions: Dimensionality of the embedding vector.
    """

    chunk_id: str
    vector: list[float]
    model: str = ""
    dimensions: int = 0

    def __post_init__(self) -> None:
        """Set dimensions from vector length if not provided."""
        if self.dimensions == 0 and self.vector:
            self.dimensions = len(self.vector)

    @property
    def norm(self) -> float:
        """Compute the L2 norm of the embedding vector.

        Returns:
            The Euclidean norm of the vector.
        """
        return float(sum(x * x for x in self.vector) ** 0.5)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"Embedding(chunk_id={self.chunk_id!r}, dims={self.dimensions}, model={self.model!r})"
        )


@dataclass
class EmbeddedChunk:
    """A chunk together with its embedding — ready for indexing.

    Attributes:
        chunk: The text chunk.
        embedding: The corresponding embedding vector.
    """

    chunk: Chunk
    embedding: Embedding


@dataclass
class SearchResult:
    """A single search result returned by the retrieval pipeline.

    Attributes:
        chunk: The matched chunk.
        score: Relevance score (higher is more relevant).
        rank: Position in the result list (1-indexed).
        metadata: Additional result-level metadata.
    """

    chunk: Chunk
    score: float
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, chunk={self.chunk.id!r})"


@dataclass
class SearchResponse:
    """A complete search response containing multiple results.

    Attributes:
        query: The original query string.
        results: Ordered list of search results.
        total_results: Total number of matching results (before limit).
        latency_ms: Search latency in milliseconds.
    """

    query: str
    results: list[SearchResult] = field(default_factory=list)
    total_results: int = 0
    latency_ms: float = 0.0

    @property
    def top_result(self) -> SearchResult | None:
        """Return the highest-scoring result, or None if empty.

        Returns:
            The top search result or None.
        """
        return self.results[0] if self.results else None

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"SearchResponse(query={self.query!r}, "
            f"results={len(self.results)}, total={self.total_results})"
        )
