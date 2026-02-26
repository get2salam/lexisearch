"""Plain-text and Markdown file loader."""

from __future__ import annotations

from pathlib import Path

from lexisearch.ingest.base import BaseLoader
from lexisearch.models import Document, DocumentFormat


class TextLoader(BaseLoader):
    """Loader for plain-text (``.txt``) and Markdown (``.md``) files.

    Reads the file content as UTF-8 and produces a single
    :class:`~lexisearch.models.Document` per file.

    Args:
        encoding: Character encoding used to read files. Defaults to
            ``"utf-8"``.
        default_metadata: Optional metadata merged into every document.

    Example:
        >>> loader = TextLoader()
        >>> docs = loader.load("notes.txt")
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        default_metadata: dict[str, object] | None = None,
    ) -> None:
        """Initialize the text loader.

        Args:
            encoding: File encoding. Defaults to ``"utf-8"``.
            default_metadata: Optional default metadata.
        """
        super().__init__(default_metadata=default_metadata)  # type: ignore[arg-type]
        self.encoding = encoding

    def load(self, source: str | Path) -> list[Document]:
        """Load a single text or Markdown file.

        Args:
            source: Path to the file.

        Returns:
            A list containing one :class:`Document`.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not self.can_load(path):
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported: .txt, .md"
            )

        content = path.read_text(encoding=self.encoding)

        fmt = (
            DocumentFormat.MARKDOWN
            if path.suffix.lower() == ".md"
            else DocumentFormat.TEXT
        )

        metadata = self._build_metadata(
            source=path,
            fmt=fmt,
            title=path.stem,
        )

        return [Document(content=content, metadata=metadata)]

    def load_from_string(
        self,
        text: str,
        source: str = "<string>",
        title: str = "",
    ) -> list[Document]:
        """Create a document directly from a string.

        Args:
            text: The document content.
            source: An identifier for the source. Defaults to ``"<string>"``.
            title: Optional title for the document.

        Returns:
            A list containing one :class:`Document`.
        """
        metadata = self._build_metadata(
            source=source,
            fmt=DocumentFormat.TEXT,
            title=title,
        )
        return [Document(content=text, metadata=metadata)]

    def supported_formats(self) -> list[DocumentFormat]:
        """Return supported formats.

        Returns:
            ``[DocumentFormat.TEXT, DocumentFormat.MARKDOWN]``
        """
        return [DocumentFormat.TEXT, DocumentFormat.MARKDOWN]
