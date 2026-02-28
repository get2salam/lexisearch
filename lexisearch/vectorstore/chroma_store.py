"""ChromaDB-backed vector store with collection management.

`ChromaDB <https://www.trychroma.com/>`_ is an open-source embedding database
with built-in persistence and metadata filtering.  This module provides a
thin adapter that maps LexiSearch's :class:`BaseVectorStore` interface onto
the ``chromadb`` Python client.

Install via ``pip install chromadb``.

Example:
    >>> from lexisearch.vectorstore.chroma_store import ChromaVectorStore
    >>> store = ChromaVectorStore(persist_directory="./chroma_data")
    >>> store.initialize()
"""

from __future__ import annotations

import logging
from typing import Any

from lexisearch.models import (
    Chunk,
    ChunkStrategy,
    EmbeddedChunk,
    Embedding,
    SearchResult,
)
from lexisearch.vectorstore.base import BaseVectorStore, DistanceMetric, VectorStoreConfig

logger = logging.getLogger(__name__)

try:
    import chromadb

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


def _require_chroma() -> None:
    """Raise a clear error when chromadb is not installed."""
    if not HAS_CHROMA:
        raise ImportError(
            "chromadb is required for ChromaVectorStore. Install with: pip install chromadb"
        )


_METRIC_MAP: dict[DistanceMetric, str] = {
    DistanceMetric.COSINE: "cosine",
    DistanceMetric.EUCLIDEAN: "l2",
    DistanceMetric.DOT_PRODUCT: "ip",
}


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed vector store.

    Supports both in-memory and persistent (on-disk) ChromaDB clients.

    Args:
        config: Store configuration.
        persist_directory: Path for on-disk persistence.  ``None`` for
            in-memory mode.
        client: Pre-configured ChromaDB client (overrides *persist_directory*).

    Raises:
        ImportError: If ``chromadb`` is not installed.
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        *,
        persist_directory: str | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize ChromaVectorStore."""
        _require_chroma()
        super().__init__(config)
        self._persist_directory = persist_directory
        self._external_client = client
        self._client: Any | None = None
        self._collection: Any | None = None
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create or connect to the ChromaDB collection."""
        _require_chroma()

        if self._external_client is not None:
            self._client = self._external_client
        elif self._persist_directory:
            self._client = chromadb.PersistentClient(
                path=self._persist_directory,
            )
        else:
            self._client = chromadb.Client()

        chroma_metric = _METRIC_MAP.get(self.config.metric, "cosine")
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": chroma_metric},
        )
        self._initialized = True
        logger.info(
            "ChromaVectorStore initialized (collection=%s, metric=%s, persist=%s)",
            self.config.collection_name,
            chroma_metric,
            self._persist_directory or "in-memory",
        )

    def close(self) -> None:
        """Release the ChromaDB client."""
        self._collection = None
        self._client = None
        self._initialized = False

    def _check_initialized(self) -> None:
        if not self._initialized or self._collection is None:
            raise RuntimeError("Store not initialized. Call initialize() or use a context manager.")

    @property
    def _col(self) -> Any:
        """Return the initialized collection (non-None)."""
        assert self._collection is not None
        return self._collection

    # ------------------------------------------------------------------
    # Payload helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata(item: EmbeddedChunk) -> dict[str, Any]:
        """Build a flat metadata dict compatible with ChromaDB.

        ChromaDB metadata values must be str, int, float, or bool.
        Nested dicts are flattened with a ``meta_`` prefix.
        """
        meta: dict[str, Any] = {
            "document_id": item.chunk.document_id,
            "chunk_index": item.chunk.index,
            "start_char": item.chunk.start_char,
            "end_char": item.chunk.end_char,
            "strategy": item.chunk.strategy.value,
            "embedding_model": item.embedding.model,
        }
        # Flatten user metadata
        for key, value in item.chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta[f"meta_{key}"] = value
        return meta

    @staticmethod
    def _reconstruct_chunk(
        chunk_id: str,
        content: str,
        metadata: dict[str, Any],
        vector: list[float] | None = None,
    ) -> EmbeddedChunk:
        """Reconstruct an EmbeddedChunk from ChromaDB data."""
        # Extract user metadata
        user_meta: dict[str, Any] = {}
        for key, value in metadata.items():
            if key.startswith("meta_"):
                user_meta[key[5:]] = value

        chunk = Chunk(
            content=content,
            document_id=metadata.get("document_id", ""),
            index=metadata.get("chunk_index", 0),
            start_char=metadata.get("start_char", 0),
            end_char=metadata.get("end_char", 0),
            metadata=user_meta,
            strategy=ChunkStrategy(metadata.get("strategy", "fixed_size")),
            id=chunk_id,
        )
        embedding = Embedding(
            chunk_id=chunk_id,
            vector=vector or [],
            model=metadata.get("embedding_model", ""),
        )
        return EmbeddedChunk(chunk=chunk, embedding=embedding)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, items: list[EmbeddedChunk]) -> list[str]:
        """Add embedded chunks to ChromaDB.

        Args:
            items: Chunks with embeddings.

        Returns:
            List of chunk IDs.
        """
        self._check_initialized()
        if not items:
            return []

        ids = [item.chunk.id for item in items]
        embeddings = [item.embedding.vector for item in items]
        documents = [item.chunk.content for item in items]
        metadatas = [self._build_metadata(item) for item in items]

        self._col.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("Added %d items to ChromaDB", len(ids))
        return ids

    def upsert(self, items: list[EmbeddedChunk]) -> list[str]:
        """Insert or update chunks in ChromaDB.

        Args:
            items: Chunks with embeddings.

        Returns:
            List of upserted IDs.
        """
        self._check_initialized()
        if not items:
            return []

        ids = [item.chunk.id for item in items]
        embeddings = [item.embedding.vector for item in items]
        documents = [item.chunk.content for item in items]
        metadatas = [self._build_metadata(item) for item in items]

        self._col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return ids

    def delete(self, ids: list[str]) -> int:
        """Remove items from ChromaDB.

        Args:
            ids: Chunk IDs to delete.

        Returns:
            Number of items removed.
        """
        self._check_initialized()
        if not ids:
            return 0

        # ChromaDB delete doesn't report count, so check existence first
        existing = self._col.get(ids=ids)
        existing_ids = existing.get("ids", [])
        if not existing_ids:
            return 0

        self._col.delete(ids=existing_ids)
        return len(existing_ids)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, id: str) -> EmbeddedChunk | None:
        """Retrieve a single item by ID.

        Args:
            id: The chunk ID.

        Returns:
            The stored chunk or ``None``.
        """
        self._check_initialized()
        result = self._col.get(
            ids=[id],
            include=["embeddings", "documents", "metadatas"],
        )
        if not result["ids"]:
            return None

        return self._reconstruct_chunk(
            chunk_id=result["ids"][0],
            content=result["documents"][0],
            metadata=result["metadatas"][0],
            vector=result["embeddings"][0] if result.get("embeddings") else None,
        )

    def list_ids(self) -> list[str]:
        """Return all stored IDs.

        Returns:
            Sorted list of chunk IDs.
        """
        self._check_initialized()
        result = self._col.get(include=[])
        return sorted(result.get("ids", []))

    def count(self) -> int:
        """Return the number of items in the collection.

        Returns:
            Item count.
        """
        if not self._initialized or self._collection is None:
            return 0
        return int(self._collection.count())

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _build_where_filter(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """Convert LexiSearch filters to ChromaDB where clause."""
        if not filters:
            return None

        conditions: list[dict[str, Any]] = []
        for key, value in filters.items():
            if key == "document_id":
                conditions.append({"document_id": {"$eq": value}})
            else:
                conditions.append({f"meta_{key}": {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Similarity search using ChromaDB.

        Args:
            query_vector: Query embedding.
            top_k: Maximum results.
            filters: Metadata filters.

        Returns:
            Ordered search results.
        """
        self._check_initialized()
        if self._col.count() == 0:
            return []

        where = self._build_where_filter(filters) if filters else None
        query_params: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": min(top_k, self._col.count()),
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if where:
            query_params["where"] = where

        raw = self._col.query(**query_params)

        results: list[SearchResult] = []
        ids_list = raw.get("ids", [[]])[0]
        docs_list = raw.get("documents", [[]])[0]
        metas_list = raw.get("metadatas", [[]])[0]
        dists_list = raw.get("distances", [[]])[0]

        for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids_list, docs_list, metas_list, dists_list, strict=False), start=1
        ):
            # ChromaDB returns distances; convert to similarity
            # For cosine: distance = 1 - similarity
            # For L2: score = 1 / (1 + distance)
            # For IP: distance = -similarity (negate)
            if self.config.metric == DistanceMetric.COSINE:
                score = 1.0 - dist
            elif self.config.metric == DistanceMetric.EUCLIDEAN:
                score = 1.0 / (1.0 + dist)
            else:  # DOT_PRODUCT
                score = -dist

            ec = self._reconstruct_chunk(cid, doc, meta)
            results.append(SearchResult(chunk=ec.chunk, score=score, rank=rank))

        return results

    def search_by_text(
        self,
        query: str,
        embedder: Any,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed a query and search ChromaDB.

        Args:
            query: Text query.
            embedder: Embedder with ``embed_text`` method.
            top_k: Maximum results.
            filters: Metadata filters.

        Returns:
            Ordered search results.
        """
        query_vector = embedder.embed_text(query)
        return self.search(query_vector, top_k=top_k, filters=filters)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, path: str | None = None) -> None:
        """Persist ChromaDB data.

        For :class:`PersistentClient`, data is auto-persisted. This method
        is a no-op but satisfies the interface.

        Args:
            path: Ignored (ChromaDB manages its own storage path).
        """
        self._check_initialized()
        # ChromaDB PersistentClient auto-persists
        logger.info("ChromaDB persist called (auto-persisted by PersistentClient)")

    def load(self, path: str) -> None:
        """Load a ChromaDB store from a persistence directory.

        Initializes a PersistentClient at *path* and opens the configured
        collection.

        Args:
            path: Path to the ChromaDB persistence directory.
        """
        _require_chroma()
        self._persist_directory = path
        self._external_client = None
        self.initialize()
