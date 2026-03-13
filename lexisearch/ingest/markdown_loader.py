"""Dedicated Markdown document loader with frontmatter and section splitting.

Unlike the plain :class:`TextLoader`, this loader parses YAML-style
frontmatter and can split a single Markdown file into per-section documents,
one per heading.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from lexisearch.ingest.base import BaseLoader
from lexisearch.models import Document, DocumentFormat


class MarkdownLoader(BaseLoader):
    """Loader for Markdown (``.md``) files with frontmatter and section support.

    Features:

    - **Frontmatter** — Parses ``---`` delimited YAML-style header blocks to
      extract ``title``, ``author``, ``date``, ``tags``, and arbitrary fields.
    - **Section splitting** — When ``split_sections=True``, produces one
      :class:`~lexisearch.models.Document` per heading block instead of a
      single document for the whole file.
    - **Image stripping** — Optionally removes ``![alt](url)`` image tags from
      the content.

    Args:
        split_sections: Split the file into per-heading documents.
        min_section_chars: Minimum character length for a section to be
            included when splitting. Shorter sections are merged with the
            next section or skipped.
        strip_images: Remove Markdown image tags before returning content.
        encoding: File encoding. Defaults to ``"utf-8"``.
        default_metadata: Loader-level default metadata.

    Example:
        >>> loader = MarkdownLoader(split_sections=True, min_section_chars=50)
        >>> docs = loader.load("guide.md")
        >>> [d.metadata.title for d in docs]
        ['Introduction', 'Usage', 'API Reference']
    """

    _FRONTMATTER_RE = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.DOTALL)
    _HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*$", re.MULTILINE)
    _IMAGE_RE = re.compile(r"!\[.*?\]\(.*?\)")

    def __init__(
        self,
        split_sections: bool = False,
        min_section_chars: int = 80,
        strip_images: bool = False,
        encoding: str = "utf-8",
        default_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Markdown loader.

        Args:
            split_sections: Produce one document per heading block.
            min_section_chars: Skip sections shorter than this when splitting.
            strip_images: Remove image tags from content.
            encoding: File encoding.
            default_metadata: Optional default metadata.
        """
        super().__init__(default_metadata=default_metadata)
        self.split_sections = split_sections
        self.min_section_chars = min_section_chars
        self.strip_images = strip_images
        self.encoding = encoding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, source: str | Path) -> list[Document]:
        """Load a Markdown file into one or more documents.

        Args:
            source: Path to a ``.md`` file.

        Returns:
            A list of :class:`Document` objects. If ``split_sections`` is
            enabled, returns one document per section; otherwise one document.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not ``.md``.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() != ".md":
            raise ValueError(f"Expected a .md file, got: {path.suffix!r}")

        raw = path.read_text(encoding=self.encoding)
        frontmatter, body = self._parse_frontmatter(raw)

        if self.strip_images:
            body = self._IMAGE_RE.sub("", body)

        title: Any = frontmatter.get("title", path.stem)
        author = str(frontmatter.get("author", ""))
        extra_fm = {k: v for k, v in frontmatter.items() if k not in {"title", "author"}}

        if self.split_sections:
            return self._split_into_sections(path, body, str(title), author, extra_fm)

        metadata = self._build_metadata(
            source=path,
            fmt=DocumentFormat.MARKDOWN,
            title=str(title),
            author=author,
            extra=extra_fm,
        )
        return [Document(content=body.strip(), metadata=metadata)]

    def load_from_string(
        self,
        text: str,
        source: str = "<string>",
    ) -> list[Document]:
        """Load Markdown content from a string.

        Args:
            text: Raw Markdown text.
            source: Source identifier.

        Returns:
            One or more :class:`Document` objects.
        """
        frontmatter, body = self._parse_frontmatter(text)
        if self.strip_images:
            body = self._IMAGE_RE.sub("", body)

        title: Any = frontmatter.get("title", source)
        author = str(frontmatter.get("author", ""))
        extra_fm = {k: v for k, v in frontmatter.items() if k not in {"title", "author"}}

        if self.split_sections:
            return self._split_into_sections(
                Path(source), body, str(title), author, extra_fm
            )

        metadata = self._build_metadata(
            source=source,
            fmt=DocumentFormat.MARKDOWN,
            title=str(title),
            author=author,
            extra=extra_fm,
        )
        return [Document(content=body.strip(), metadata=metadata)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        """Extract and parse YAML-style frontmatter from text.

        Args:
            text: Full Markdown file content.

        Returns:
            A ``(frontmatter_dict, body)`` tuple.
        """
        match = self._FRONTMATTER_RE.match(text)
        if not match:
            return {}, text

        fm_block = match.group(1)
        body = text[match.end():]

        frontmatter: dict[str, Any] = {}
        for line in fm_block.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                frontmatter[key.strip()] = val.strip()

        return frontmatter, body

    def _split_into_sections(
        self,
        path: Path,
        body: str,
        doc_title: str,
        author: str,
        extra_fm: dict[str, Any],
    ) -> list[Document]:
        """Split body into per-section documents.

        Args:
            path: Source path (used for source references in metadata).
            body: Markdown body text (frontmatter stripped).
            doc_title: Document-level title from frontmatter or filename.
            author: Author from frontmatter.
            extra_fm: Extra frontmatter fields.

        Returns:
            A list of :class:`Document` objects, one per heading block.
        """
        heading_matches = list(self._HEADING_RE.finditer(body))
        if not heading_matches:
            metadata = self._build_metadata(
                source=path,
                fmt=DocumentFormat.MARKDOWN,
                title=doc_title,
                author=author,
                extra=extra_fm,
            )
            return [Document(content=body.strip(), metadata=metadata)]

        documents: list[Document] = []
        for i, match in enumerate(heading_matches):
            start = match.start()
            end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(body)
            section_text = body[start:end].strip()

            if len(section_text) < self.min_section_chars:
                continue

            level = len(match.group(1))
            heading = match.group(2).strip()

            extra: dict[str, Any] = {
                "section_level": level,
                "section_index": i,
                "doc_title": doc_title,
                **extra_fm,
            }
            metadata = self._build_metadata(
                source=f"{path}#section{i}",
                fmt=DocumentFormat.MARKDOWN,
                title=heading,
                author=author,
                extra=extra,
            )
            documents.append(Document(content=section_text, metadata=metadata))

        if not documents:
            metadata = self._build_metadata(
                source=path,
                fmt=DocumentFormat.MARKDOWN,
                title=doc_title,
                author=author,
                extra=extra_fm,
            )
            return [Document(content=body.strip(), metadata=metadata)]

        return documents

    def supported_formats(self) -> list[DocumentFormat]:
        """Return supported formats.

        Returns:
            ``[DocumentFormat.MARKDOWN]``
        """
        return [DocumentFormat.MARKDOWN]
