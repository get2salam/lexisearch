"""HTML document loader using BeautifulSoup.

Requires the ``html`` extra: ``pip install lexisearch[html]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lexisearch.ingest.base import BaseLoader
from lexisearch.models import Document, DocumentFormat

_BS4_AVAILABLE: bool
try:
    from bs4 import BeautifulSoup  # type: ignore[import-untyped]

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


class HTMLLoader(BaseLoader):
    """Loader for HTML documents via BeautifulSoup.

    Strips HTML tags and extracts visible text content.  Optionally
    preserves the ``<title>`` element as document title metadata.

    Args:
        parser: BeautifulSoup parser backend (default ``"html.parser"``).
        strip_tags: HTML tags whose content should be removed entirely
            (e.g., ``["script", "style"]``).
        default_metadata: Optional metadata merged into every document.

    Raises:
        ImportError: If BeautifulSoup is not installed.

    Example:
        >>> loader = HTMLLoader()
        >>> docs = loader.load("page.html")
    """

    DEFAULT_STRIP_TAGS: list[str] = ["script", "style", "nav", "footer", "header"]

    def __init__(
        self,
        parser: str = "html.parser",
        strip_tags: list[str] | None = None,
        default_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the HTML loader.

        Args:
            parser: BeautifulSoup parser name.
            strip_tags: Tags to remove completely. Uses
                :data:`DEFAULT_STRIP_TAGS` when ``None``.
            default_metadata: Optional default metadata.

        Raises:
            ImportError: If BeautifulSoup is not installed.
        """
        if not _BS4_AVAILABLE:
            raise ImportError(
                "BeautifulSoup4 is required for HTMLLoader. "
                "Install it with: pip install lexisearch[html]"
            )
        super().__init__(default_metadata=default_metadata)
        self.parser = parser
        self.strip_tags = strip_tags if strip_tags is not None else self.DEFAULT_STRIP_TAGS

    def load(self, source: str | Path) -> list[Document]:
        """Load an HTML file and extract its text content.

        Args:
            source: Path to the HTML file.

        Returns:
            A list containing one :class:`Document`.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        raw_html = path.read_text(encoding="utf-8")
        return self.load_from_string(raw_html, source=str(path))

    def load_from_string(
        self,
        html: str,
        source: str = "<html>",
    ) -> list[Document]:
        """Create a document from raw HTML markup.

        Args:
            html: Raw HTML string.
            source: An identifier for the source.

        Returns:
            A list containing one :class:`Document`.
        """
        soup = BeautifulSoup(html, self.parser)

        # Remove unwanted tags
        for tag_name in self.strip_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()

        # Extract visible text
        text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        lines = text.splitlines()
        cleaned_lines: list[str] = []
        prev_empty = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if not prev_empty:
                    cleaned_lines.append("")
                prev_empty = True
            else:
                cleaned_lines.append(stripped)
                prev_empty = False

        content = "\n".join(cleaned_lines).strip()

        metadata = self._build_metadata(
            source=source,
            fmt=DocumentFormat.HTML,
            title=title,
        )
        return [Document(content=content, metadata=metadata)]

    def supported_formats(self) -> list[DocumentFormat]:
        """Return supported formats.

        Returns:
            ``[DocumentFormat.HTML]``
        """
        return [DocumentFormat.HTML]
