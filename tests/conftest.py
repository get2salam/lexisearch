"""Shared pytest fixtures for LexiSearch tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from lexisearch.models import (
    Chunk,
    ChunkStrategy,
    Document,
    DocumentFormat,
    DocumentMetadata,
)


@pytest.fixture
def sample_text() -> str:
    """Return a multi-paragraph sample text."""
    return (
        "Machine learning is a subset of artificial intelligence. "
        "It focuses on building systems that learn from data. "
        "Deep learning is a further subset of machine learning.\n\n"
        "Natural language processing deals with the interaction between "
        "computers and human language. It involves tasks like translation, "
        "summarization, and sentiment analysis.\n\n"
        "Information retrieval is the science of searching for information "
        "in documents. Modern search engines use dense retrieval methods "
        "based on neural embeddings."
    )


@pytest.fixture
def sample_document(sample_text: str) -> Document:
    """Return a sample Document instance."""
    return Document(
        content=sample_text,
        metadata=DocumentMetadata(
            source="test.txt",
            title="Test Document",
            format=DocumentFormat.TEXT,
        ),
    )


@pytest.fixture
def short_document() -> Document:
    """Return a short document for edge-case testing."""
    return Document(
        content="Hello world.",
        metadata=DocumentMetadata(
            source="short.txt",
            title="Short",
            format=DocumentFormat.TEXT,
        ),
    )


@pytest.fixture
def empty_document() -> Document:
    """Return a document with empty content."""
    return Document(
        content="",
        metadata=DocumentMetadata(source="empty.txt"),
    )


@pytest.fixture
def sample_chunk(sample_document: Document) -> Chunk:
    """Return a sample Chunk instance."""
    return Chunk(
        content="Machine learning is a subset of artificial intelligence.",
        document_id=sample_document.id,
        index=0,
        start_char=0,
        end_char=55,
        strategy=ChunkStrategy.FIXED_SIZE,
    )


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def txt_file(tmp_dir: Path, sample_text: str) -> Path:
    """Write sample text to a .txt file and return its path."""
    p = tmp_dir / "sample.txt"
    p.write_text(sample_text, encoding="utf-8")
    return p


@pytest.fixture
def html_file(tmp_dir: Path) -> Path:
    """Write a sample HTML file and return its path."""
    content = """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Introduction</h1>
<p>This is a test document about information retrieval.</p>
<script>var x = 1;</script>
<style>body { color: black; }</style>
<p>It covers dense retrieval and neural search.</p>
</body>
</html>"""
    p = tmp_dir / "test.html"
    p.write_text(content, encoding="utf-8")
    return p
