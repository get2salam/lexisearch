"""Sentence-based text chunker.

Groups sentences together until the target chunk size is reached.
"""

from __future__ import annotations

import re

from lexisearch.chunking.base import BaseChunker
from lexisearch.models import Chunk, ChunkStrategy, Document


class SentenceChunker(BaseChunker):
    """Split text into chunks along sentence boundaries.

    Sentences are detected using a simple regex-based splitter.  Sentences
    are accumulated until the next sentence would exceed ``chunk_size``,
    at which point a new chunk is started.

    Args:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Number of trailing characters from the previous
            chunk to prepend to the next.
        min_sentence_length: Ignore "sentences" shorter than this.

    Example:
        >>> from lexisearch.models import Document
        >>> chunker = SentenceChunker(chunk_size=200, chunk_overlap=0)
        >>> doc = Document(content="Hello world. This is a test. Another sentence.")
        >>> chunks = chunker.chunk(doc)
    """

    # Regex that splits after sentence-ending punctuation followed by
    # whitespace or end-of-string.
    _SENTENCE_PATTERN: re.Pattern[str] = re.compile(
        r"(?<=[.!?])\s+",
    )

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_sentence_length: int = 5,
    ) -> None:
        """Initialize the sentence chunker.

        Args:
            chunk_size: Target chunk size.
            chunk_overlap: Overlap between chunks.
            min_sentence_length: Minimum length for a valid sentence.
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.min_sentence_length = min_sentence_length

    def chunk(self, document: Document) -> list[Chunk]:
        """Split the document by sentences.

        Args:
            document: The document to split.

        Returns:
            Ordered list of :class:`Chunk` objects.
        """
        text = document.content
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        grouped = self._group_sentences(sentences)

        chunks: list[Chunk] = []
        offset = 0
        for idx, group_text in enumerate(grouped):
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

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        raw = self._SENTENCE_PATTERN.split(text)
        return [s.strip() for s in raw if len(s.strip()) >= self.min_sentence_length]

    def _group_sentences(self, sentences: list[str]) -> list[str]:
        """Group sentences into chunks that fit within the size limit.

        Args:
            sentences: Ordered list of sentences.

        Returns:
            List of grouped text chunks.
        """
        groups: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            added_len = len(sentence) + (1 if current else 0)

            if current_len + added_len > self.chunk_size and current:
                groups.append(" ".join(current))
                # Overlap: keep trailing sentences that fit
                overlap_sentences: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) + 1 <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break
                current = overlap_sentences
                current_len = sum(len(s) for s in current) + max(0, len(current) - 1)

            current.append(sentence)
            current_len += added_len

        if current:
            groups.append(" ".join(current))

        return groups

    def strategy(self) -> ChunkStrategy:
        """Return the sentence strategy identifier.

        Returns:
            ``ChunkStrategy.SENTENCE``
        """
        return ChunkStrategy.SENTENCE
