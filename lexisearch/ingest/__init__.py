"""Document ingestion loaders for LexiSearch.

This package provides loaders for various document formats. All loaders
implement the :class:`BaseLoader` interface, making them interchangeable
in the ingestion pipeline.

Example:
    >>> from lexisearch.ingest import TextLoader
    >>> loader = TextLoader()
    >>> docs = loader.load("example.txt")
"""

from __future__ import annotations

from lexisearch.ingest.base import BaseLoader
from lexisearch.ingest.dedup import (
    ContentHashRegistry,
    DeduplicationFilter,
    DuplicateDocumentError,
    content_hash,
)
from lexisearch.ingest.html_loader import HTMLLoader
from lexisearch.ingest.pdf_loader import PDFLoader
from lexisearch.ingest.text_loader import TextLoader

__all__ = [
    "BaseLoader",
    "ContentHashRegistry",
    "DeduplicationFilter",
    "DuplicateDocumentError",
    "HTMLLoader",
    "PDFLoader",
    "TextLoader",
    "content_hash",
]
