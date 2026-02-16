"""Tests for lexisearch.ingest loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from lexisearch.ingest.text_loader import TextLoader
from lexisearch.models import DocumentFormat


class TestTextLoader:
    """Tests for TextLoader."""

    def test_load_txt_file(self, txt_file: Path) -> None:
        """TextLoader should load a .txt file."""
        loader = TextLoader()
        docs = loader.load(txt_file)
        assert len(docs) == 1
        assert "Machine learning" in docs[0].content

    def test_load_sets_metadata(self, txt_file: Path) -> None:
        """Loaded document should have proper metadata."""
        loader = TextLoader()
        docs = loader.load(txt_file)
        meta = docs[0].metadata
        assert meta.format == DocumentFormat.TEXT
        assert meta.title == "sample"
        assert str(txt_file) in meta.source

    def test_load_missing_file(self, tmp_dir: Path) -> None:
        """Loading a missing file should raise FileNotFoundError."""
        loader = TextLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_dir / "nonexistent.txt")

    def test_load_unsupported_extension(self, tmp_dir: Path) -> None:
        """Loading a .xyz file should raise ValueError."""
        p = tmp_dir / "file.xyz"
        p.write_text("data", encoding="utf-8")
        loader = TextLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(p)

    def test_load_markdown(self, tmp_dir: Path) -> None:
        """TextLoader should handle .md files."""
        p = tmp_dir / "readme.md"
        p.write_text("# Title\n\nBody text.", encoding="utf-8")
        loader = TextLoader()
        docs = loader.load(p)
        assert docs[0].metadata.format == DocumentFormat.MARKDOWN

    def test_load_from_string(self) -> None:
        """load_from_string should create a document from raw text."""
        loader = TextLoader()
        docs = loader.load_from_string("hello world", title="Greeting")
        assert len(docs) == 1
        assert docs[0].content == "hello world"
        assert docs[0].metadata.title == "Greeting"

    def test_default_metadata_merged(self, txt_file: Path) -> None:
        """Default metadata should be merged into the extra field."""
        loader = TextLoader(default_metadata={"project": "test"})
        docs = loader.load(txt_file)
        assert docs[0].metadata.extra["project"] == "test"

    def test_supported_formats(self) -> None:
        """TextLoader should support TEXT and MARKDOWN."""
        loader = TextLoader()
        fmts = loader.supported_formats()
        assert DocumentFormat.TEXT in fmts
        assert DocumentFormat.MARKDOWN in fmts

    def test_can_load_txt(self) -> None:
        """can_load should return True for .txt files."""
        loader = TextLoader()
        assert loader.can_load("document.txt") is True

    def test_can_load_pdf_is_false(self) -> None:
        """can_load should return False for .pdf files."""
        loader = TextLoader()
        assert loader.can_load("document.pdf") is False


class TestHTMLLoader:
    """Tests for HTMLLoader (when bs4 is available)."""

    def test_load_html_file(self, html_file: Path) -> None:
        """HTMLLoader should extract text from HTML."""
        try:
            from lexisearch.ingest.html_loader import HTMLLoader
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")

        loader = HTMLLoader()
        docs = loader.load(html_file)
        assert len(docs) == 1
        assert "information retrieval" in docs[0].content
        # Script and style content should be stripped
        assert "var x" not in docs[0].content
        assert "color: black" not in docs[0].content

    def test_html_title_extraction(self, html_file: Path) -> None:
        """HTMLLoader should extract the title from <title> tag."""
        try:
            from lexisearch.ingest.html_loader import HTMLLoader
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")

        loader = HTMLLoader()
        docs = loader.load(html_file)
        assert docs[0].metadata.title == "Test Page"

    def test_html_from_string(self) -> None:
        """load_from_string should parse raw HTML."""
        try:
            from lexisearch.ingest.html_loader import HTMLLoader
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")

        loader = HTMLLoader()
        docs = loader.load_from_string("<p>Hello</p>")
        assert "Hello" in docs[0].content

    def test_html_missing_file(self, tmp_dir: Path) -> None:
        """Loading a missing HTML file should raise FileNotFoundError."""
        try:
            from lexisearch.ingest.html_loader import HTMLLoader
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")

        loader = HTMLLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_dir / "missing.html")
