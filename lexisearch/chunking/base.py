"""Abstract base class for text chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lexisearch.models import Chunk, ChunkStrategy, Document


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies.

    Subclasses must implement :meth:`chunk` and :meth:`strategy`.

    Attributes:
        chunk_size: Target number of characters (or tokens) per chunk.
        chunk_overlap: Number of overlapping characters between consecutive
            chunks.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Target chunk size in characters. Must be positive.
            chunk_overlap: Overlap between consecutive chunks. Must be
                non-negative and less than ``chunk_size``.

        Raises:
            ValueError: If constraints on size/overlap are violated.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            document: The document to chunk.

        Returns:
            An ordered list of :class:`Chunk` objects.
        """
        ...

    @abstractmethod
    def strategy(self) -> ChunkStrategy:
        """Return the chunking strategy identifier.

        Returns:
            The :class:`ChunkStrategy` enum value for this chunker.
        """
        ...

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents.

        Args:
            documents: Documents to chunk.

        Returns:
            A flat list of chunks from all documents, in order.
        """
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks

    def _build_chunk(
        self,
        content: str,
        document: Document,
        index: int,
        start_char: int,
        end_char: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        """Create a :class:`Chunk` with standard fields populated.

        Args:
            content: The chunk text.
            document: The source document.
            index: Zero-based index of the chunk within the document.
            start_char: Starting character offset.
            end_char: Ending character offset.
            extra_metadata: Additional metadata to attach.

        Returns:
            A populated :class:`Chunk`.
        """
        metadata: dict[str, Any] = {
            "source": document.metadata.source,
            "document_title": document.metadata.title,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return Chunk(
            content=content,
            document_id=document.id,
            index=index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata,
            strategy=self.strategy(),
        )
