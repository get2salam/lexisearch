"""In-memory vector store with brute-force similarity search.

This implementation stores all vectors in a plain Python dictionary and
computes similarities on every query.  It is ideal for:

- **Testing** — no external dependencies required.
- **Small datasets** — perfectly adequate for < 10 000 vectors.
- **Prototyping** — swap in FAISS / Qdrant later without changing your code.

Persistence is handled via :mod:`json`, so the store can be saved to and
loaded from a single ``.json`` file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lexisearch.models import (
    Chunk,
    ChunkStrategy,
    EmbeddedChunk,
    Embedding,
    SearchResult,
)
from lexisearch.vectorstore.base import BaseVectorStore, DistanceMetric, VectorStoreConfig
from lexisearch.vectorstore.metrics import compute_score

logger = logging.getLogger(__name__)


@dataclass
class _StoredItem:
    """Internal wrapper for a chunk + vector pair."""

    chunk_id: str
    document_id: str
    content: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    strategy: str = "fixed_size"
    embedding_model: str = ""


class InMemoryVectorStore(BaseVectorStore):
    """Pure-Python brute-force vector store.

    Stores embedded chunks in a dictionary keyed by chunk ID.  All
    similarity searches iterate over the full collection — O(n) per query.

    Args:
        config: Store configuration.  Only ``dimensions``, ``metric`` and
            ``collection_name`` are used.

    Example:
        >>> from lexisearch.vectorstore.memory import InMemoryVectorStore
        >>> store = InMemoryVectorStore()
        >>> store.initialize()
        >>> store.count()
        0
    """

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        """Initialize InMemoryVectorStore."""
        super().__init__(config)
        self._items: dict[str, _StoredItem] = {}
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Prepare the store for use."""
        self._items = {}
        self._initialized = True
        logger.info(
            "InMemoryVectorStore initialized (collection=%s, dims=%d, metric=%s)",
            self.config.collection_name,
            self.config.dimensions,
            self.config.metric.value,
        )

    def close(self) -> None:
        """Release resources (no-op for in-memory)."""
        self._initialized = False

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "Store not initialized. Call initialize() or use a context manager."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_stored(item: EmbeddedChunk) -> _StoredItem:
        """Convert an EmbeddedChunk into internal storage format."""
        return _StoredItem(
            chunk_id=item.chunk.id,
            document_id=item.chunk.document_id,
            content=item.chunk.content,
            vector=item.embedding.vector,
            metadata=item.chunk.metadata,
            chunk_index=item.chunk.index,
            start_char=item.chunk.start_char,
            end_char=item.chunk.end_char,
            strategy=item.chunk.strategy.value,
            embedding_model=item.embedding.model,
        )

    @staticmethod
    def _to_embedded_chunk(stored: _StoredItem) -> EmbeddedChunk:
        """Reconstruct an EmbeddedChunk from stored data."""
        chunk = Chunk(
            content=stored.content,
            document_id=stored.document_id,
            index=stored.chunk_index,
            start_char=stored.start_char,
            end_char=stored.end_char,
            metadata=stored.metadata,
            strategy=ChunkStrategy(stored.strategy),
            id=stored.chunk_id,
        )
        embedding = Embedding(
            chunk_id=stored.chunk_id,
            vector=stored.vector,
            model=stored.embedding_model,
        )
        return EmbeddedChunk(chunk=chunk, embedding=embedding)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, items: list[EmbeddedChunk]) -> list[str]:
        """Insert embedded chunks.  Raises on duplicate IDs.

        Args:
            items: Chunks with embeddings.

        Returns:
            List of inserted IDs.

        Raises:
            ValueError: If any chunk ID already exists.
        """
        self._check_initialized()
        ids: list[str] = []
        for item in items:
            cid = item.chunk.id
            if cid in self._items:
                raise ValueError(f"Duplicate chunk ID: {cid}")
            self._validate_dimensions(item.embedding.vector)
            self._items[cid] = self._to_stored(item)
            ids.append(cid)
        logger.debug("Added %d items (total=%d)", len(ids), len(self._items))
        return ids

    def upsert(self, items: list[EmbeddedChunk]) -> list[str]:
        """Insert or replace embedded chunks.

        Args:
            items: Chunks with embeddings.

        Returns:
            List of upserted IDs.
        """
        self._check_initialized()
        ids: list[str] = []
        for item in items:
            cid = item.chunk.id
            self._validate_dimensions(item.embedding.vector)
            self._items[cid] = self._to_stored(item)
            ids.append(cid)
        return ids

    def delete(self, ids: list[str]) -> int:
        """Remove items by ID.

        Args:
            ids: Chunk IDs to remove.

        Returns:
            Number of items removed.
        """
        self._check_initialized()
        removed = 0
        for cid in ids:
            if cid in self._items:
                del self._items[cid]
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, id: str) -> EmbeddedChunk | None:
        """Retrieve a chunk by ID.

        Args:
            id: The chunk ID.

        Returns:
            The stored chunk or ``None``.
        """
        self._check_initialized()
        stored = self._items.get(id)
        if stored is None:
            return None
        return self._to_embedded_chunk(stored)

    def list_ids(self) -> list[str]:
        """Return all stored IDs in sorted order.

        Returns:
            Sorted list of chunk IDs.
        """
        self._check_initialized()
        return sorted(self._items.keys())

    def count(self) -> int:
        """Return the number of stored items.

        Returns:
            Item count.
        """
        if not self._initialized:
            return 0
        return len(self._items)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Brute-force similarity search.

        Iterates all items, computes similarity, and returns the top-k.

        Args:
            query_vector: The query embedding.
            top_k: Maximum number of results.
            filters: Metadata key-value pairs; all must match.

        Returns:
            Ordered list of :class:`SearchResult`.
        """
        self._check_initialized()
        self._validate_dimensions(query_vector)

        scored: list[tuple[float, _StoredItem]] = []
        for stored in self._items.values():
            if filters and not self._matches_filters(stored, filters):
                continue
            score = compute_score(
                query_vector, stored.vector, self.config.metric
            )
            scored.append((score, stored))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[SearchResult] = []
        for rank, (score, stored) in enumerate(scored[:top_k], start=1):
            ec = self._to_embedded_chunk(stored)
            results.append(
                SearchResult(chunk=ec.chunk, score=score, rank=rank)
            )
        return results

    def search_by_text(
        self,
        query: str,
        embedder: Any,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed a query and perform similarity search.

        Args:
            query: Natural-language query string.
            embedder: Embedder with an ``embed_text`` method.
            top_k: Maximum results.
            filters: Metadata filters.

        Returns:
            Ordered search results.
        """
        query_vector = embedder.embed_text(query)
        return self.search(query_vector, top_k=top_k, filters=filters)

    # ------------------------------------------------------------------
    # Persistence (JSON)
    # ------------------------------------------------------------------

    def persist(self, path: str | None = None) -> None:
        """Save the store to a JSON file.

        Args:
            path: File path.  Defaults to ``<collection_name>.json``.
        """
        self._check_initialized()
        out_path = Path(path or f"{self.config.collection_name}.json")
        data = {
            "collection_name": self.config.collection_name,
            "dimensions": self.config.dimensions,
            "metric": self.config.metric.value,
            "items": {
                cid: {
                    "chunk_id": s.chunk_id,
                    "document_id": s.document_id,
                    "content": s.content,
                    "vector": s.vector,
                    "metadata": s.metadata,
                    "chunk_index": s.chunk_index,
                    "start_char": s.start_char,
                    "end_char": s.end_char,
                    "strategy": s.strategy,
                    "embedding_model": s.embedding_model,
                }
                for cid, s in self._items.items()
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Persisted %d items to %s", len(self._items), out_path)

    def load(self, path: str) -> None:
        """Load items from a JSON file.

        Args:
            path: Path to a previously persisted file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Store file not found: {path}")

        raw = json.loads(file_path.read_text(encoding="utf-8"))

        self.config.collection_name = raw.get(
            "collection_name", self.config.collection_name
        )
        self.config.dimensions = raw.get("dimensions", self.config.dimensions)
        self.config.metric = DistanceMetric(
            raw.get("metric", self.config.metric.value)
        )

        self._items.clear()
        for cid, item_data in raw.get("items", {}).items():
            self._items[cid] = _StoredItem(
                chunk_id=item_data["chunk_id"],
                document_id=item_data["document_id"],
                content=item_data["content"],
                vector=item_data["vector"],
                metadata=item_data.get("metadata", {}),
                chunk_index=item_data.get("chunk_index", 0),
                start_char=item_data.get("start_char", 0),
                end_char=item_data.get("end_char", 0),
                strategy=item_data.get("strategy", "fixed_size"),
                embedding_model=item_data.get("embedding_model", ""),
            )

        self._initialized = True
        logger.info("Loaded %d items from %s", len(self._items), path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dimensions(self, vector: list[float]) -> None:
        """Raise if vector dimensionality doesn't match config."""
        if len(vector) != self.config.dimensions:
            raise ValueError(
                f"Expected {self.config.dimensions}-dim vector, "
                f"got {len(vector)}"
            )

    @staticmethod
    def _matches_filters(stored: _StoredItem, filters: dict[str, Any]) -> bool:
        """Check whether a stored item matches all metadata filters."""
        for key, value in filters.items():
            # Support top-level attributes and metadata dict
            if key == "document_id":
                if stored.document_id != value:
                    return False
            elif key not in stored.metadata or stored.metadata[key] != value:
                return False
        return True
