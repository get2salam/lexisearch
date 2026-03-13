"""CSV and TSV document loader.

Each row in the file becomes a :class:`~lexisearch.models.Document`.
The content is formed from specified columns or from all columns joined.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from lexisearch.ingest.base import BaseLoader
from lexisearch.models import Document, DocumentFormat


class CSVLoader(BaseLoader):
    r"""Loader for CSV (``.csv``) and TSV (``.tsv``) files.

    Reads the file with :mod:`csv.DictReader`. Each data row is converted
    to one :class:`~lexisearch.models.Document`. The ``content_columns``
    parameter controls which columns form the document body; if omitted all
    columns are joined as ``key: value`` pairs.

    Args:
        content_columns: Column names whose values form the document content.
        title_column: Column name used as the document title.
        delimiter: Field delimiter. Defaults to ``","``; overridden to
            ``"\t"`` automatically for ``.tsv`` files.
        encoding: File encoding. Defaults to ``"utf-8"``.
        include_header_in_extra: Attach the raw row dict to document metadata.
        default_metadata: Loader-level default metadata.

    Example:
        >>> loader = CSVLoader(content_columns=["abstract"], title_column="title")
        >>> docs = loader.load("papers.csv")
    """

    def __init__(
        self,
        content_columns: list[str] | None = None,
        title_column: str | None = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        include_header_in_extra: bool = True,
        default_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the CSV loader.

        Args:
            content_columns: Columns to use as content. All if omitted.
            title_column: Column used as document title.
            delimiter: Column separator character.
            encoding: File encoding.
            include_header_in_extra: Store entire row in metadata ``extra``.
            default_metadata: Optional default metadata.
        """
        super().__init__(default_metadata=default_metadata)
        self.content_columns = content_columns
        self.title_column = title_column
        self.delimiter = delimiter
        self.encoding = encoding
        self.include_header_in_extra = include_header_in_extra

    def load(self, source: str | Path) -> list[Document]:
        """Load documents from a CSV or TSV file.

        Args:
            source: Path to the file.

        Returns:
            A list of :class:`Document` objects, one per data row.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the extension is unsupported.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in {".csv", ".tsv"}:
            raise ValueError(f"Unsupported extension: {path.suffix!r}. Expected .csv or .tsv")

        delimiter = "\t" if path.suffix.lower() == ".tsv" else self.delimiter

        documents: list[Document] = []
        with path.open(encoding=self.encoding, newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for i, row in enumerate(reader):
                content = self._extract_content(row)
                title = (
                    str(row[self.title_column])
                    if self.title_column and self.title_column in row
                    else f"{path.stem}[{i}]"
                )
                extra: dict[str, Any] = dict(row) if self.include_header_in_extra else {}

                metadata = self._build_metadata(
                    source=f"{path}#row{i}",
                    fmt=DocumentFormat.CSV,
                    title=title,
                    extra=extra,
                )
                documents.append(Document(content=content, metadata=metadata))

        return documents

    def load_from_rows(
        self,
        rows: list[dict[str, str]],
        source: str = "<memory>",
    ) -> list[Document]:
        """Create documents directly from an in-memory list of row dicts.

        Args:
            rows: List of ``{column: value}`` dicts.
            source: Identifier string for the source.

        Returns:
            A list of :class:`Document` objects.
        """
        documents: list[Document] = []
        for i, row in enumerate(rows):
            content = self._extract_content(row)
            title = (
                str(row[self.title_column])
                if self.title_column and self.title_column in row
                else f"row[{i}]"
            )
            extra: dict[str, Any] = dict(row) if self.include_header_in_extra else {}
            metadata = self._build_metadata(
                source=f"{source}#row{i}",
                fmt=DocumentFormat.CSV,
                title=title,
                extra=extra,
            )
            documents.append(Document(content=content, metadata=metadata))
        return documents

    def _extract_content(self, row: dict[str, str]) -> str:
        if self.content_columns:
            parts = [row[col] for col in self.content_columns if col in row]
            return "\n".join(parts)
        return " | ".join(f"{k}: {v}" for k, v in row.items())

    def supported_formats(self) -> list[DocumentFormat]:
        """Return supported formats.

        Returns:
            ``[DocumentFormat.CSV]``
        """
        return [DocumentFormat.CSV]
