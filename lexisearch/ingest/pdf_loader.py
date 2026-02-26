"""PDF document loader using PyMuPDF (fitz).

Requires the ``pdf`` extra: ``pip install lexisearch[pdf]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lexisearch.ingest.base import BaseLoader
from lexisearch.models import Document, DocumentFormat

_PYMUPDF_AVAILABLE: bool
try:
    import fitz  # type: ignore[import-untyped]

    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False


class PDFLoader(BaseLoader):
    """Loader for PDF documents via PyMuPDF.

    Extracts text from each page and concatenates into a single document.
    Optionally, each page can be returned as a separate document.

    Args:
        per_page: If ``True``, emit one :class:`Document` per page.
        default_metadata: Optional metadata merged into every document.

    Raises:
        ImportError: If PyMuPDF is not installed.

    Example:
        >>> loader = PDFLoader()
        >>> docs = loader.load("report.pdf")
    """

    def __init__(
        self,
        per_page: bool = False,
        default_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the PDF loader.

        Args:
            per_page: Emit one document per page when ``True``.
            default_metadata: Optional default metadata.

        Raises:
            ImportError: If PyMuPDF is not installed.
        """
        if not _PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF is required for PDFLoader. Install it with: pip install lexisearch[pdf]"
            )
        super().__init__(default_metadata=default_metadata)
        self.per_page = per_page

    def load(self, source: str | Path) -> list[Document]:
        """Load a PDF file.

        Args:
            source: Path to the PDF file.

        Returns:
            A list of :class:`Document` objects. One document if
            ``per_page`` is ``False``; one per page otherwise.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        pdf = fitz.open(str(path))  # type: ignore[union-attr]
        try:
            page_count = len(pdf)

            if self.per_page:
                return self._load_per_page(pdf, path, page_count)
            return self._load_full(pdf, path, page_count)
        finally:
            pdf.close()

    def _load_full(
        self,
        pdf: Any,
        path: Path,
        page_count: int,
    ) -> list[Document]:
        """Extract all pages into a single document."""
        pages: list[str] = []
        for page in pdf:
            text: str = page.get_text()
            if text.strip():
                pages.append(text)

        content = "\n\n".join(pages)
        metadata = self._build_metadata(
            source=path,
            fmt=DocumentFormat.PDF,
            title=path.stem,
            page_count=page_count,
        )
        return [Document(content=content, metadata=metadata)]

    def _load_per_page(
        self,
        pdf: Any,
        path: Path,
        page_count: int,
    ) -> list[Document]:
        """Extract each page as a separate document."""
        documents: list[Document] = []
        for i, page in enumerate(pdf):
            text: str = page.get_text()
            if not text.strip():
                continue

            metadata = self._build_metadata(
                source=path,
                fmt=DocumentFormat.PDF,
                title=f"{path.stem} — Page {i + 1}",
                page_count=page_count,
                extra={"page_number": i + 1},
            )
            documents.append(Document(content=text, metadata=metadata))

        return documents

    def supported_formats(self) -> list[DocumentFormat]:
        """Return supported formats.

        Returns:
            ``[DocumentFormat.PDF]``
        """
        return [DocumentFormat.PDF]
