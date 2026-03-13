"""Tests for extended document loaders: JSONLoader, CSVLoader, MarkdownLoader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from lexisearch.ingest import CSVLoader, JSONLoader, MarkdownLoader
from lexisearch.models import DocumentFormat

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# JSONLoader tests
# ---------------------------------------------------------------------------


class TestJSONLoader:
    def test_load_array_no_content_key(self, tmp_dir: Path) -> None:
        data = [{"title": "Doc A", "text": "Alpha"}, {"title": "Doc B", "text": "Beta"}]
        p = tmp_dir / "docs.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        loader = JSONLoader()
        docs = loader.load(p)
        assert len(docs) == 2
        assert "Alpha" in docs[0].content

    def test_load_array_with_content_key(self, tmp_dir: Path) -> None:
        data = [{"id": 1, "body": "Hello world"}, {"id": 2, "body": "Goodbye world"}]
        p = tmp_dir / "items.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        loader = JSONLoader(content_key="body", title_key="id")
        docs = loader.load(p)
        assert docs[0].content == "Hello world"
        assert docs[1].content == "Goodbye world"

    def test_load_single_object(self, tmp_dir: Path) -> None:
        data = {"title": "Single", "body": "One document only"}
        p = tmp_dir / "single.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        loader = JSONLoader(content_key="body")
        docs = loader.load(p)
        assert len(docs) == 1
        assert docs[0].content == "One document only"

    def test_load_jsonl(self, tmp_dir: Path) -> None:
        lines = [
            json.dumps({"text": "Line one"}),
            json.dumps({"text": "Line two"}),
            json.dumps({"text": "Line three"}),
        ]
        p = tmp_dir / "data.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        loader = JSONLoader(content_key="text")
        docs = loader.load(p)
        assert len(docs) == 3
        assert docs[2].content == "Line three"

    def test_jsonl_skips_blank_lines(self, tmp_dir: Path) -> None:
        p = tmp_dir / "sparse.jsonl"
        p.write_text('{"x": 1}\n\n{"x": 2}\n', encoding="utf-8")
        docs = JSONLoader().load(p)
        assert len(docs) == 2

    def test_metadata_keys_extracted(self, tmp_dir: Path) -> None:
        data = [{"text": "Content", "author": "Alice", "year": 2024}]
        p = tmp_dir / "meta.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        loader = JSONLoader(content_key="text", metadata_keys=["author", "year"])
        docs = loader.load(p)
        assert docs[0].metadata.extra["author"] == "Alice"
        assert docs[0].metadata.extra["year"] == 2024

    def test_format_is_json(self, tmp_dir: Path) -> None:
        p = tmp_dir / "f.json"
        p.write_text('[{"k": "v"}]', encoding="utf-8")
        docs = JSONLoader().load(p)
        assert docs[0].metadata.format == DocumentFormat.JSON

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            JSONLoader().load("/nonexistent/path.json")

    def test_unsupported_extension_raises(self, tmp_dir: Path) -> None:
        p = tmp_dir / "bad.txt"
        p.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported extension"):
            JSONLoader().load(p)

    def test_load_from_records(self) -> None:
        records = [{"body": "Alpha"}, {"body": "Beta"}]
        loader = JSONLoader(content_key="body")
        docs = loader.load_from_records(records, source="test")
        assert len(docs) == 2
        assert docs[0].content == "Alpha"

    def test_supported_formats(self) -> None:
        assert DocumentFormat.JSON in JSONLoader().supported_formats()

    def test_can_load(self, tmp_dir: Path) -> None:
        p = tmp_dir / "x.json"
        p.touch()
        assert JSONLoader().can_load(p)
        assert not JSONLoader().can_load(tmp_dir / "x.pdf")

    def test_empty_array(self, tmp_dir: Path) -> None:
        p = tmp_dir / "empty.json"
        p.write_text("[]", encoding="utf-8")
        docs = JSONLoader().load(p)
        assert docs == []

    def test_title_key_used_as_title(self, tmp_dir: Path) -> None:
        data = [{"label": "My Title", "content": "Body text here"}]
        p = tmp_dir / "titled.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        loader = JSONLoader(content_key="content", title_key="label")
        docs = loader.load(p)
        assert docs[0].metadata.title == "My Title"

    def test_non_dict_records_stringified(self, tmp_dir: Path) -> None:
        p = tmp_dir / "strings.json"
        p.write_text('["alpha", "beta", "gamma"]', encoding="utf-8")
        docs = JSONLoader().load(p)
        assert len(docs) == 3
        assert docs[0].content == "alpha"


# ---------------------------------------------------------------------------
# CSVLoader tests
# ---------------------------------------------------------------------------


class TestCSVLoader:
    def _write_csv(self, path: Path, rows: list[str]) -> None:
        path.write_text("\n".join(rows), encoding="utf-8")

    def test_basic_load(self, tmp_dir: Path) -> None:
        p = tmp_dir / "data.csv"
        self._write_csv(p, ["title,body", "Doc A,Content A", "Doc B,Content B"])
        loader = CSVLoader(content_columns=["body"], title_column="title")
        docs = loader.load(p)
        assert len(docs) == 2
        assert docs[0].content == "Content A"
        assert docs[0].metadata.title == "Doc A"

    def test_all_columns_joined_when_no_content_columns(self, tmp_dir: Path) -> None:
        p = tmp_dir / "all.csv"
        self._write_csv(p, ["name,role", "Alice,Engineer", "Bob,Designer"])
        docs = CSVLoader().load(p)
        assert "name: Alice" in docs[0].content
        assert "role: Engineer" in docs[0].content

    def test_tsv_auto_detect(self, tmp_dir: Path) -> None:
        p = tmp_dir / "data.tsv"
        p.write_text("col1\tcol2\nval1\tval2\n", encoding="utf-8")
        docs = CSVLoader(content_columns=["col1"]).load(p)
        assert len(docs) == 1
        assert docs[0].content == "val1"

    def test_format_is_csv(self, tmp_dir: Path) -> None:
        p = tmp_dir / "f.csv"
        self._write_csv(p, ["a,b", "1,2"])
        docs = CSVLoader().load(p)
        assert docs[0].metadata.format == DocumentFormat.CSV

    def test_extra_contains_row(self, tmp_dir: Path) -> None:
        p = tmp_dir / "rows.csv"
        self._write_csv(p, ["x,y", "10,20"])
        docs = CSVLoader(include_header_in_extra=True).load(p)
        assert docs[0].metadata.extra["x"] == "10"

    def test_no_extra_when_disabled(self, tmp_dir: Path) -> None:
        p = tmp_dir / "rows.csv"
        self._write_csv(p, ["x,y", "10,20"])
        docs = CSVLoader(include_header_in_extra=False).load(p)
        assert docs[0].metadata.extra == {}

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            CSVLoader().load("/missing.csv")

    def test_unsupported_extension_raises(self, tmp_dir: Path) -> None:
        p = tmp_dir / "bad.md"
        p.write_text("a,b\n1,2", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported extension"):
            CSVLoader().load(p)

    def test_empty_file_returns_empty_list(self, tmp_dir: Path) -> None:
        p = tmp_dir / "empty.csv"
        self._write_csv(p, ["col"])  # header only, no data
        docs = CSVLoader().load(p)
        assert docs == []

    def test_multiple_content_columns(self, tmp_dir: Path) -> None:
        p = tmp_dir / "multi.csv"
        self._write_csv(p, ["title,abstract,keywords", "RAG,Retrieval paper,search nlp"])
        docs = CSVLoader(content_columns=["abstract", "keywords"]).load(p)
        assert "Retrieval paper" in docs[0].content
        assert "search nlp" in docs[0].content

    def test_load_from_rows(self) -> None:
        rows = [{"text": "Alpha", "id": "1"}, {"text": "Beta", "id": "2"}]
        docs = CSVLoader(content_columns=["text"]).load_from_rows(rows)
        assert docs[0].content == "Alpha"

    def test_supported_formats(self) -> None:
        assert DocumentFormat.CSV in CSVLoader().supported_formats()

    def test_can_load_tsv(self, tmp_dir: Path) -> None:
        p = tmp_dir / "x.tsv"
        p.touch()
        assert CSVLoader().can_load(p)

    def test_source_includes_row_index(self, tmp_dir: Path) -> None:
        p = tmp_dir / "idx.csv"
        self._write_csv(p, ["col", "val0", "val1"])
        docs = CSVLoader().load(p)
        assert "row0" in docs[0].metadata.source
        assert "row1" in docs[1].metadata.source


# ---------------------------------------------------------------------------
# MarkdownLoader tests
# ---------------------------------------------------------------------------

MD_WITH_FRONTMATTER = """\
---
title: My Guide
author: Alice
date: 2024-01-15
---

# Introduction

This section introduces the topic. It provides essential background
information for readers who are new to the subject matter.

## Getting Started

Follow these steps to get up and running quickly with the framework.
You can install it via pip and start using it in minutes.

## Advanced Usage

Once you are comfortable with the basics, explore advanced features
like custom retrievers, pipeline composition, and plugin hooks.
"""

MD_NO_FRONTMATTER = """\
# Plain Heading

Some content here with no frontmatter present.

## Sub-section

More content in a sub-section without any YAML header block.
"""


class TestMarkdownLoader:
    def test_load_without_split(self, tmp_dir: Path) -> None:
        p = tmp_dir / "guide.md"
        p.write_text(MD_WITH_FRONTMATTER, encoding="utf-8")
        docs = MarkdownLoader().load(p)
        assert len(docs) == 1
        assert "Introduction" in docs[0].content

    def test_frontmatter_extracted(self, tmp_dir: Path) -> None:
        p = tmp_dir / "guide.md"
        p.write_text(MD_WITH_FRONTMATTER, encoding="utf-8")
        docs = MarkdownLoader().load(p)
        assert docs[0].metadata.title == "My Guide"
        assert docs[0].metadata.author == "Alice"

    def test_frontmatter_extra_fields(self, tmp_dir: Path) -> None:
        p = tmp_dir / "guide.md"
        p.write_text(MD_WITH_FRONTMATTER, encoding="utf-8")
        docs = MarkdownLoader().load(p)
        assert docs[0].metadata.extra.get("date") == "2024-01-15"

    def test_split_sections(self, tmp_dir: Path) -> None:
        p = tmp_dir / "guide.md"
        p.write_text(MD_WITH_FRONTMATTER, encoding="utf-8")
        docs = MarkdownLoader(split_sections=True, min_section_chars=10).load(p)
        titles = [d.metadata.title for d in docs]
        assert "Introduction" in titles
        assert "Getting Started" in titles
        assert "Advanced Usage" in titles

    def test_split_sections_metadata(self, tmp_dir: Path) -> None:
        p = tmp_dir / "guide.md"
        p.write_text(MD_WITH_FRONTMATTER, encoding="utf-8")
        docs = MarkdownLoader(split_sections=True, min_section_chars=10).load(p)
        for doc in docs:
            assert "section_level" in doc.metadata.extra
            assert "doc_title" in doc.metadata.extra
            assert doc.metadata.extra["doc_title"] == "My Guide"

    def test_no_frontmatter_uses_filename(self, tmp_dir: Path) -> None:
        p = tmp_dir / "plain.md"
        p.write_text(MD_NO_FRONTMATTER, encoding="utf-8")
        docs = MarkdownLoader().load(p)
        assert docs[0].metadata.title == "plain"

    def test_split_no_frontmatter(self, tmp_dir: Path) -> None:
        p = tmp_dir / "plain.md"
        p.write_text(MD_NO_FRONTMATTER, encoding="utf-8")
        docs = MarkdownLoader(split_sections=True, min_section_chars=10).load(p)
        assert len(docs) >= 1
        assert any("Plain Heading" in d.metadata.title for d in docs)

    def test_strip_images(self, tmp_dir: Path) -> None:
        content = "# Title\n\nSome text ![alt](image.png) more text.\n"
        p = tmp_dir / "img.md"
        p.write_text(content, encoding="utf-8")
        docs = MarkdownLoader(strip_images=True).load(p)
        assert "![alt]" not in docs[0].content

    def test_images_kept_by_default(self, tmp_dir: Path) -> None:
        content = "# Title\n\nText ![alt](image.png) here.\n"
        p = tmp_dir / "img.md"
        p.write_text(content, encoding="utf-8")
        docs = MarkdownLoader().load(p)
        assert "![alt]" in docs[0].content

    def test_format_is_markdown(self, tmp_dir: Path) -> None:
        p = tmp_dir / "x.md"
        p.write_text("# Hello\nWorld", encoding="utf-8")
        docs = MarkdownLoader().load(p)
        assert docs[0].metadata.format == DocumentFormat.MARKDOWN

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            MarkdownLoader().load("/no/such/file.md")

    def test_wrong_extension_raises(self, tmp_dir: Path) -> None:
        p = tmp_dir / "bad.txt"
        p.write_text("# Hi", encoding="utf-8")
        with pytest.raises(ValueError, match=r"Expected a \.md file"):
            MarkdownLoader().load(p)

    def test_min_section_chars_filter(self, tmp_dir: Path) -> None:
        content = "# Long Section\n\n" + "x" * 200 + "\n\n# Short\n\nTiny.\n"
        p = tmp_dir / "filter.md"
        p.write_text(content, encoding="utf-8")
        docs = MarkdownLoader(split_sections=True, min_section_chars=100).load(p)
        titles = [d.metadata.title for d in docs]
        assert "Long Section" in titles
        assert "Short" not in titles

    def test_load_from_string(self) -> None:
        docs = MarkdownLoader().load_from_string("# Hello\n\nWorld content.", source="inline")
        assert len(docs) == 1
        assert "World content" in docs[0].content

    def test_supported_formats(self) -> None:
        assert DocumentFormat.MARKDOWN in MarkdownLoader().supported_formats()

    def test_can_load(self, tmp_dir: Path) -> None:
        p = tmp_dir / "x.md"
        p.touch()
        assert MarkdownLoader().can_load(p)
        assert not MarkdownLoader().can_load(tmp_dir / "x.pdf")

    def test_load_from_string_with_split(self) -> None:
        md = "# Section One\n\nContent of section one here.\n\n## Section Two\n\nContent two.\n"
        docs = MarkdownLoader(split_sections=True, min_section_chars=5).load_from_string(md)
        titles = [d.metadata.title for d in docs]
        assert "Section One" in titles
