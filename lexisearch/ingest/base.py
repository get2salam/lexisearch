"""Abstract base class for document loaders.

All document loaders in LexiSearch must implement the :class:`BaseLoader`
interface.  This ensures a consistent API across different file formats
and ingestion sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from lexisearch.models import Document, DocumentFormat, DocumentMetadata


class BaseLoader(ABC):
    """Abstract base class for document loaders.

    Subclasses must implement :meth:`load` and :meth:`supported_formats`.

    Attributes:
        default_metadata: Default metadata merged into every loaded document.
    """

    def __init__(self, default_metadata: dict[str, Any] | None = None) -> None:
        """Initialize the loader with optional default metadata.

        Args:
            default_metadata: Key-value pairs merged into every document's
                metadata ``extra`` field.
        """
        self.default_metadata: dict[str, Any] = default_metadata or {}

    @abstractmethod
    def load(self, source: str | Path) -> list[Document]:
        """Load documents from the given source.

        Args:
            source: A file path, URL, or other identifier understood by the
                concrete loader.

        Returns:
            A list of :class:`Document` objects extracted from the source.

        Raises:
            FileNotFoundError: If the source file does not exist.
            ValueError: If the source format is not supported.
        """
        ...

    @abstractmethod
    def supported_formats(self) -> list[DocumentFormat]:
        """Return the document formats this loader can handle.

        Returns:
            A list of supported :class:`DocumentFormat` values.
        """
        ...

    def can_load(self, source: str | Path) -> bool:
        """Check whether this loader can handle the given source.

        The default implementation checks the file extension against
        supported formats.  Subclasses may override for smarter detection.

        Args:
            source: The source path or identifier.

        Returns:
            True if the loader can handle this source.
        """
        path = Path(source)
        ext_to_format: dict[str, DocumentFormat] = {
            ".txt": DocumentFormat.TEXT,
            ".md": DocumentFormat.MARKDOWN,
            ".pdf": DocumentFormat.PDF,
            ".html": DocumentFormat.HTML,
            ".htm": DocumentFormat.HTML,
        }
        fmt = ext_to_format.get(path.suffix.lower(), DocumentFormat.UNKNOWN)
        return fmt in self.supported_formats()

    def _build_metadata(
        self,
        source: str | Path,
        fmt: DocumentFormat,
        **kwargs: Any,
    ) -> DocumentMetadata:
        """Build a :class:`DocumentMetadata` instance for a loaded document.

        Merges loader-level defaults with per-document overrides.

        Args:
            source: The source path or identifier.
            fmt: The document format.
            **kwargs: Additional fields forwarded to :class:`DocumentMetadata`.

        Returns:
            A populated metadata instance.
        """
        extra = {**self.default_metadata, **kwargs.pop("extra", {})}
        return DocumentMetadata(
            source=str(source),
            format=fmt,
            extra=extra,
            **kwargs,
        )
