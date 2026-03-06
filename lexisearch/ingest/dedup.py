"""Content-hash deduplication for the ingestion pipeline.

``DeduplicationFilter`` wraps any document loader and transparently skips
documents whose content has been seen before.  De-duplication is based on
a SHA-256 digest of the normalised document content, making it robust to
minor whitespace differences while detecting identical documents.

Usage::

    from lexisearch.ingest import TextLoader
    from lexisearch.ingest.dedup import DeduplicationFilter

    loader = TextLoader()
    dedup = DeduplicationFilter(loader)

    docs = dedup.load_many(["paper.txt", "paper_copy.txt"])
    # paper_copy.txt is skipped if its content matches paper.txt

    print(dedup.stats())
    # {'total_seen': 2, 'duplicates_skipped': 1, 'unique_loaded': 1}

The filter can also be used standalone, without a wrapped loader::

    from lexisearch.ingest.dedup import DeduplicationFilter, ContentHashRegistry

    registry = ContentHashRegistry()
    is_new, digest = registry.register(document)
    if is_new:
        pipeline.ingest(document)

Thread safety
-------------
Both ``ContentHashRegistry`` and ``DeduplicationFilter`` are thread-safe
when the GIL is held (standard CPython dict updates are atomic for single
keys).  For multi-process scenarios, use a shared-memory backend (e.g.
Redis) and wire up a custom ``ContentHashRegistry`` subclass.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lexisearch.models import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Content normalisation
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Collapse whitespace and lowercase for a stable hash."""
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def content_hash(document: Document) -> str:
    """Return the SHA-256 hex digest of the normalised document content.

    The hash is computed over the lowercased, whitespace-normalised content
    so that trivial formatting differences (e.g. trailing newlines, double
    spaces) do not result in duplicate entries.

    Parameters
    ----------
    document:
        The document whose content will be hashed.

    Returns:
    -------
    str
        64-character hex digest.
    """
    normalised = _normalise(document.content)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class ContentHashRegistry:
    """In-memory registry of seen content hashes.

    Maintains a set of SHA-256 digests for documents that have been
    registered.  Thread-safe under the GIL.

    Attributes:
    ----------
    seen:
        Set of hex digests already processed.
    """

    seen: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def register(self, document: Document) -> tuple[bool, str]:
        """Register a document and return ``(is_new, digest)``.

        Parameters
        ----------
        document:
            The document to register.

        Returns:
        -------
        tuple[bool, str]
            ``(True, digest)`` if this is a new document;
            ``(False, digest)`` if it has been seen before.
        """
        digest = content_hash(document)
        with self._lock:
            if digest in self.seen:
                return False, digest
            self.seen.add(digest)
            return True, digest

    def seen_count(self) -> int:
        """Return the number of distinct content hashes registered."""
        return len(self.seen)

    def reset(self) -> None:
        """Clear all registered hashes (useful between pipeline runs)."""
        with self._lock:
            self.seen.clear()

    def contains(self, document: Document) -> bool:
        """Return ``True`` if the document's hash is already registered."""
        return content_hash(document) in self.seen

    def add_digest(self, digest: str) -> None:
        """Manually register a pre-computed digest."""
        with self._lock:
            self.seen.add(digest)


# ---------------------------------------------------------------------------
# Deduplication filter
# ---------------------------------------------------------------------------


class DeduplicationFilter:
    """Wraps a document loader and de-duplicates based on content hash.

    Parameters
    ----------
    loader:
        The underlying loader to delegate to.  Must implement ``load(path)``
        returning a ``Document`` or ``list[Document]``.
    registry:
        An existing ``ContentHashRegistry`` to share across multiple filters.
        If ``None``, a new registry is created for this filter instance.
    strategy:
        ``"skip"`` (default) — silently skip duplicate documents.
        ``"warn"``           — log a warning and skip.
        ``"raise"``          — raise ``DuplicateDocumentError``.
    """

    def __init__(
        self,
        loader: Any = None,
        *,
        registry: ContentHashRegistry | None = None,
        strategy: str = "skip",
    ) -> None:
        """Initialise the deduplication filter.

        Args:
            loader: Underlying document loader (optional).
            registry: Shared hash registry (creates a new one if ``None``).
            strategy: One of ``"skip"``, ``"warn"``, or ``"raise"``.
        """
        if strategy not in ("skip", "warn", "raise"):
            raise ValueError(
                f"Invalid dedup strategy: {strategy!r}. Use 'skip', 'warn', or 'raise'."
            )

        self._loader = loader
        self._registry = registry or ContentHashRegistry()
        self._strategy = strategy

        self._total_seen: int = 0
        self._duplicates_skipped: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core filter API
    # ------------------------------------------------------------------

    def is_duplicate(self, document: Document) -> bool:
        """Return ``True`` if this document's content has been seen before."""
        return self._registry.contains(document)

    def filter(self, documents: list[Document]) -> list[Document]:
        """Return only the unique documents from *documents*.

        Side-effects: updates the registry with each unique document's hash.

        Parameters
        ----------
        documents:
            Input list, potentially containing duplicates.

        Returns:
        -------
        list[Document]
            Filtered list with only first-seen documents, preserving order.
        """
        unique: list[Document] = []
        for doc in documents:
            is_new, digest = self._registry.register(doc)
            with self._lock:
                self._total_seen += 1
            if is_new:
                unique.append(doc)
            else:
                with self._lock:
                    self._duplicates_skipped += 1
                msg = (
                    f"Duplicate document skipped (hash={digest[:12]}…, "
                    f"id={doc.id!r}, title={getattr(doc, 'title', '')!r})"
                )
                if self._strategy == "warn":
                    logger.warning(msg)
                elif self._strategy == "raise":
                    raise DuplicateDocumentError(msg, digest=digest, document=doc)
                else:
                    logger.debug(msg)
        return unique

    # ------------------------------------------------------------------
    # Loader delegation
    # ------------------------------------------------------------------

    def load(self, path: str) -> Document | list[Document]:
        """Load a document (or documents) from *path* and filter duplicates.

        Delegates to the wrapped loader, then applies deduplication.

        Parameters
        ----------
        path:
            File path or URL passed to the underlying loader.

        Returns:
        -------
        Document | list[Document]
            A single document or list, with duplicates removed.
        """
        if self._loader is None:
            raise RuntimeError("No loader configured. Pass a loader to DeduplicationFilter().")

        raw = self._loader.load(path)
        if isinstance(raw, list):
            return self.filter(raw)
        # Single document
        filtered = self.filter([raw])
        return filtered[0] if filtered else raw  # return original if deduped (caller decides)

    def load_many(self, paths: list[str]) -> list[Document]:
        """Load and deduplicate documents from multiple *paths*.

        Parameters
        ----------
        paths:
            List of file paths or URLs.

        Returns:
        -------
        list[Document]
            All unique documents across all paths, in discovery order.
        """
        if self._loader is None:
            raise RuntimeError("No loader configured.")

        all_docs: list[Document] = []
        for path in paths:
            raw = self._loader.load(path)
            docs = raw if isinstance(raw, list) else [raw]
            all_docs.extend(docs)

        return self.filter(all_docs)

    # ------------------------------------------------------------------
    # Stats & state management
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return ingestion statistics.

        Returns:
        -------
        dict[str, int]
            ``total_seen`` — total documents processed (including duplicates).
            ``duplicates_skipped`` — documents discarded as duplicates.
            ``unique_loaded`` — distinct documents passed through.
            ``registry_size`` — number of hashes in the registry.
        """
        with self._lock:
            total = self._total_seen
            dupes = self._duplicates_skipped
        return {
            "total_seen": total,
            "duplicates_skipped": dupes,
            "unique_loaded": total - dupes,
            "registry_size": self._registry.seen_count(),
        }

    def reset(self) -> None:
        """Reset statistics and clear the hash registry."""
        with self._lock:
            self._total_seen = 0
            self._duplicates_skipped = 0
        self._registry.reset()


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class DuplicateDocumentError(Exception):
    """Raised when a duplicate document is encountered and strategy='raise'."""

    def __init__(self, message: str, digest: str = "", document: Document | None = None) -> None:
        """Initialise with a message, optional digest and document."""
        super().__init__(message)
        self.digest = digest
        self.document = document
