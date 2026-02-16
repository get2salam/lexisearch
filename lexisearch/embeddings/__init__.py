"""Embedding providers for LexiSearch.

This package contains embedding backends that convert text chunks into
dense vector representations.  All embedders implement the
:class:`BaseEmbedder` interface.

Example:
    >>> from lexisearch.embeddings import MockEmbedder
    >>> embedder = MockEmbedder(dimensions=384)
"""

from __future__ import annotations

from lexisearch.embeddings.base import BaseEmbedder
from lexisearch.embeddings.mock import MockEmbedder
from lexisearch.embeddings.openai_embedder import OpenAIEmbedder
from lexisearch.embeddings.sbert import SentenceTransformerEmbedder

__all__ = [
    "BaseEmbedder",
    "MockEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
]
