"""JSON and JSONL document loader.

Supports:
- Single JSON objects → one document
- JSON arrays → one document per item
- JSONL (newline-delimited JSON) → one document per line
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lexisearch.ingest.base import BaseLoader
from lexisearch.models import Document, DocumentFormat


class JSONLoader(BaseLoader):
    """Loader for JSON (``.json``) and JSONL (``.jsonl``) files.

    Handles single objects, arrays, and newline-delimited JSON. A configurable
    ``content_key`` extracts the primary text field from each record; if
    omitted the full record is JSON-serialized as the document content.

    Args:
        content_key: Dict key whose value becomes the document content.
        title_key: Dict key used as the document title.
        metadata_keys: Additional dict keys promoted to document metadata.
        default_metadata: Loader-level default metadata.

    Example:
        >>> loader = JSONLoader(content_key="body", title_key="title")
        >>> docs = loader.load("articles.json")
    """

    def __init__(
        self,
        content_key: str | None = None,
        title_key: str | None = None,
        metadata_keys: list[str] | None = None,
        default_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the JSON loader.

        Args:
            content_key: Key whose value is used as document content.
            title_key: Key used as document title.
            metadata_keys: Extra keys surfaced in document metadata.
            default_metadata: Optional default metadata merged into every doc.
        """
        super().__init__(default_metadata=default_metadata)
        self.content_key = content_key
        self.title_key = title_key
        self.metadata_keys: list[str] = metadata_keys or []

    def load(self, source: str | Path) -> list[Document]:
        """Load documents from a JSON or JSONL file.

        Args:
            source: Path to a ``.json`` or ``.jsonl`` file.

        Returns:
            A list of :class:`Document` objects, one per record.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is unsupported.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported extension: {path.suffix!r}. Expected .json or .jsonl")

        records = (
            self._load_jsonl(path) if path.suffix.lower() == ".jsonl" else self._load_json(path)
        )

        documents: list[Document] = []
        for i, record in enumerate(records):
            if isinstance(record, dict):
                content = self._extract_content(record)
                title = (
                    str(record[self.title_key])
                    if self.title_key and self.title_key in record
                    else f"{path.stem}[{i}]"
                )
                extra = {k: record[k] for k in self.metadata_keys if k in record}
            else:
                content = str(record)
                title = f"{path.stem}[{i}]"
                extra = {}

            metadata = self._build_metadata(
                source=f"{path}#{i}",
                fmt=DocumentFormat.JSON,
                title=title,
                extra=extra,
            )
            documents.append(Document(content=content, metadata=metadata))

        return documents

    def load_from_records(
        self,
        records: list[Any],
        source: str = "<memory>",
    ) -> list[Document]:
        """Create documents directly from an in-memory list of records.

        Args:
            records: A list of dicts or strings to convert to documents.
            source: Identifier string for the source.

        Returns:
            A list of :class:`Document` objects.
        """
        documents: list[Document] = []
        for i, record in enumerate(records):
            if isinstance(record, dict):
                content = self._extract_content(record)
                title = (
                    str(record[self.title_key])
                    if self.title_key and self.title_key in record
                    else f"record[{i}]"
                )
                extra = {k: record[k] for k in self.metadata_keys if k in record}
            else:
                content = str(record)
                title = f"record[{i}]"
                extra = {}

            metadata = self._build_metadata(
                source=f"{source}#{i}",
                fmt=DocumentFormat.JSON,
                title=title,
                extra=extra,
            )
            documents.append(Document(content=content, metadata=metadata))

        return documents

    def _extract_content(self, record: dict[str, Any]) -> str:
        if self.content_key and self.content_key in record:
            return str(record[self.content_key])
        return json.dumps(record, ensure_ascii=False, indent=2)

    def _load_json(self, path: Path) -> list[Any]:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return [data]

    def _load_jsonl(self, path: Path) -> list[Any]:
        records: list[Any] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
        return records

    def supported_formats(self) -> list[DocumentFormat]:
        """Return supported formats.

        Returns:
            ``[DocumentFormat.JSON]``
        """
        return [DocumentFormat.JSON]
