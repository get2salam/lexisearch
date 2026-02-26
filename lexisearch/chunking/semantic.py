"""Semantic text chunker.

Groups sentences by estimated semantic similarity using a simple
heuristic (shared-word overlap).  For production use, plug in a real
embedding model via the ``similarity_fn`` parameter.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from lexisearch.chunking.base import BaseChunker
from lexisearch.models import Chunk, ChunkStrategy, Document

if TYPE_CHECKING:
    from collections.abc import Callable


def _default_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings' word sets.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Jaccard coefficient in ``[0, 1]``.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class SemanticChunker(BaseChunker):
    """Split text into semantically coherent chunks.

    Sentences are grouped together as long as consecutive sentences remain
    similar above a configurable threshold.  When similarity drops, a new
    chunk is started.

    By default, similarity is estimated with Jaccard word overlap.  You can
    replace this with an embedding-based similarity function for better
    results.

    Args:
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Character overlap between chunks.
        similarity_threshold: Minimum similarity to keep sentences in the
            same chunk (default ``0.3``).
        similarity_fn: A callable ``(str, str) -> float`` that returns a
            similarity score in ``[0, 1]``.

    Example:
        >>> from lexisearch.models import Document
        >>> chunker = SemanticChunker(chunk_size=500, similarity_threshold=0.2)
        >>> doc = Document(content="Cats are cute. Dogs are loyal. Python is great.")
        >>> chunks = chunker.chunk(doc)
    """

    _SENTENCE_SPLIT: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.3,
        similarity_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        """Initialize the semantic chunker.

        Args:
            chunk_size: Maximum chunk size.
            chunk_overlap: Overlap between chunks.
            similarity_threshold: Similarity cutoff for grouping.
            similarity_fn: Custom similarity function.
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self.similarity_fn: Callable[[str, str], float] = (
            similarity_fn or _default_similarity
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split the document into semantically coherent chunks.

        Args:
            document: The document to split.

        Returns:
            Ordered list of :class:`Chunk` objects.
        """
        text = document.content
        if not text.strip():
            return []

        sentences = [s.strip() for s in self._SENTENCE_SPLIT.split(text) if s.strip()]
        if not sentences:
            return [
                self._build_chunk(
                    content=text,
                    document=document,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                )
            ]

        groups = self._group_by_similarity(sentences)

        chunks: list[Chunk] = []
        offset = 0
        for idx, group_text in enumerate(groups):
            start = text.find(group_text, offset)
            if start == -1:
                start = offset
            end = start + len(group_text)

            chunks.append(
                self._build_chunk(
                    content=group_text,
                    document=document,
                    index=idx,
                    start_char=start,
                    end_char=end,
                )
            )
            offset = max(offset, start + 1)

        return chunks

    def _group_by_similarity(self, sentences: list[str]) -> list[str]:
        """Group sentences by semantic similarity.

        Args:
            sentences: Ordered list of sentences.

        Returns:
            List of grouped text chunks.
        """
        groups: list[str] = []
        current: list[str] = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            prev_sentence = sentences[i - 1]

            added_len = len(sentence) + 1
            size_ok = current_len + added_len <= self.chunk_size
            sim = self.similarity_fn(prev_sentence, sentence)
            similar = sim >= self.similarity_threshold

            if size_ok and similar:
                current.append(sentence)
                current_len += added_len
            else:
                groups.append(" ".join(current))
                current = [sentence]
                current_len = len(sentence)

        if current:
            groups.append(" ".join(current))

        return groups

    def strategy(self) -> ChunkStrategy:
        """Return the semantic strategy identifier.

        Returns:
            ``ChunkStrategy.SEMANTIC``
        """
        return ChunkStrategy.SEMANTIC
