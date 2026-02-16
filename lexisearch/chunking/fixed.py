"""Fixed-size text chunker."""

from __future__ import annotations

from lexisearch.chunking.base import BaseChunker
from lexisearch.models import Chunk, ChunkStrategy, Document


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size character chunks with optional overlap.

    This is the simplest chunking strategy: the text is divided into
    segments of exactly ``chunk_size`` characters (the last chunk may be
    shorter).  Consecutive chunks share ``chunk_overlap`` characters.

    Args:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Character overlap between consecutive chunks.

    Example:
        >>> from lexisearch.models import Document
        >>> chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        >>> doc = Document(content="x" * 250)
        >>> chunks = chunker.chunk(doc)
        >>> len(chunks)
        3
    """

    def chunk(self, document: Document) -> list[Chunk]:
        """Split the document into fixed-size chunks.

        Args:
            document: The document to split.

        Returns:
            Ordered list of :class:`Chunk` objects.
        """
        text = document.content
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    self._build_chunk(
                        content=chunk_text,
                        document=document,
                        index=index,
                        start_char=start,
                        end_char=end,
                    )
                )
                index += 1

            start += step

        return chunks

    def strategy(self) -> ChunkStrategy:
        """Return the fixed-size strategy identifier.

        Returns:
            ``ChunkStrategy.FIXED_SIZE``
        """
        return ChunkStrategy.FIXED_SIZE
