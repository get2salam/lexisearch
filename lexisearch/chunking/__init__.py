"""Text chunking strategies for LexiSearch.

This package provides multiple chunking strategies that split documents
into smaller, retrieval-friendly segments.  All chunkers implement the
:class:`BaseChunker` interface.

Example:
    >>> from lexisearch.chunking import RecursiveChunker
    >>> chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
"""

from __future__ import annotations

from lexisearch.chunking.base import BaseChunker
from lexisearch.chunking.fixed import FixedSizeChunker
from lexisearch.chunking.recursive import RecursiveChunker
from lexisearch.chunking.semantic import SemanticChunker
from lexisearch.chunking.sentence import SentenceChunker

__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceChunker",
]
