"""Recursive character text chunker.

Splits text using a hierarchy of separators, recursively subdividing
until each piece fits within the target chunk size.
"""

from __future__ import annotations

from typing import ClassVar

from lexisearch.chunking.base import BaseChunker
from lexisearch.models import Chunk, ChunkStrategy, Document


class RecursiveChunker(BaseChunker):
    r"""Recursively split text using a hierarchy of separators.

    Tries the first separator; for any piece still larger than
    ``chunk_size``, the next separator is tried, and so on.  Falls back
    to character-level splitting as a last resort.

    Args:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Character overlap between consecutive chunks.
        separators: Ordered list of separator strings to try.
            Defaults to ``["\\n\\n", "\\n", ". ", " ", ""]``.

    Example:
        >>> from lexisearch.models import Document
        >>> chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        >>> doc = Document(content="Para one.\\n\\nPara two.\\n\\nPara three.")
        >>> chunks = chunker.chunk(doc)
    """

    DEFAULT_SEPARATORS: ClassVar[list[str]] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the recursive chunker.

        Args:
            chunk_size: Target chunk size.
            chunk_overlap: Overlap between chunks.
            separators: Custom separator hierarchy.
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        """Split the document recursively.

        Args:
            document: The document to split.

        Returns:
            Ordered list of :class:`Chunk` objects.
        """
        text = document.content
        if not text.strip():
            return []

        pieces = self._split_recursive(text, self.separators)
        merged = self._merge_pieces(pieces)

        chunks: list[Chunk] = []
        offset = 0
        for idx, piece in enumerate(merged):
            start = text.find(piece, offset)
            if start == -1:
                start = offset
            end = start + len(piece)

            chunks.append(
                self._build_chunk(
                    content=piece,
                    document=document,
                    index=idx,
                    start_char=start,
                    end_char=end,
                )
            )
            offset = max(offset, start + 1)

        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text by trying separators in order.

        Args:
            text: Text to split.
            separators: Remaining separators to try.

        Returns:
            List of text pieces, each ≤ ``chunk_size``.
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Hard split at chunk_size
            result: list[str] = []
            for i in range(0, len(text), self.chunk_size):
                piece = text[i : i + self.chunk_size]
                if piece.strip():
                    result.append(piece)
            return result

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return self._split_recursive(text, remaining_separators)

        parts = text.split(separator)

        good_pieces: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{separator}{part}" if current else part

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    good_pieces.append(current)
                if len(part) <= self.chunk_size:
                    current = part
                else:
                    # Recurse with next separator
                    sub_pieces = self._split_recursive(part, remaining_separators)
                    good_pieces.extend(sub_pieces)
                    current = ""

        if current.strip():
            good_pieces.append(current)

        return good_pieces

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """Merge small pieces and apply overlap.

        Args:
            pieces: Text pieces to merge.

        Returns:
            Merged pieces with overlap applied.
        """
        if not pieces:
            return []

        merged: list[str] = []
        for piece in pieces:
            if merged and len(merged[-1]) + len(piece) + 1 <= self.chunk_size:
                merged[-1] = f"{merged[-1]} {piece}"
            else:
                merged.append(piece)

        if self.chunk_overlap > 0 and len(merged) > 1:
            overlapped: list[str] = [merged[0]]
            for i in range(1, len(merged)):
                prev = merged[i - 1]
                overlap_text = prev[-self.chunk_overlap :]
                combined = f"{overlap_text} {merged[i]}"
                if len(combined) <= self.chunk_size:
                    overlapped.append(combined)
                else:
                    overlapped.append(merged[i])
            return overlapped

        return merged

    def strategy(self) -> ChunkStrategy:
        """Return the recursive strategy identifier.

        Returns:
            ``ChunkStrategy.RECURSIVE``
        """
        return ChunkStrategy.RECURSIVE
