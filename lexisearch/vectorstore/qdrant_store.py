"""Qdrant-backed vector store with payload filtering.

`Qdrant <https://qdrant.tech/>`_ is a high-performance vector database
with rich payload filtering, snapshot support, and horizontal scaling.
This module wraps the ``qdrant-client`` Python SDK behind the
:class:`BaseVectorStore` interface.

Supports three connection modes:

- **In-memory** — for testing (no server required).
- **Local disk** — on-disk persistence via Qdrant's embedded mode.
- **Remote** — connecting to a running Qdrant server.

Install via ``pip install qdrant-client``.

Example:
    >>> from lexisearch.vectorstore.qdrant_store import QdrantVectorStore
    >>> store = QdrantVectorStore(location=":memory:", dimensions=384)
    >>> store.initialize()
"""

from __future__ import annotations

import logging
import uuid
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
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


def _require_qdrant() -> None:
    """Raise a clear error when qdrant-client is not installed."""
    if not HAS_QDRANT:
        raise ImportError(
            "qdrant-client is required for QdrantVectorStore. "
            "Install with: pip install qdrant-client"
        )


_DISTANCE_MAP: dict[DistanceMetric, Any] = {}
if HAS_QDRANT:
    _DISTANCE_MAP = {
        DistanceMetric.COSINE: Distance.COSINE,
        DistanceMetric.EUCLIDEAN: Distance.EUCLID,
        DistanceMetric.DOT_PRODUCT: Distance.DOT,
    }


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-backed vector store.

    Args:
        config: Store configuration.
        location: Qdrant connection string:
            - ``":memory:"`` for in-memory mode.
            - A local directory path for on-disk embedded mode.
            - ``"http://host:port"`` for a remote Qdrant server.
        api_key: Optional API key for authenticated remote connections.
        dimensions: Vector dimensionality (convenience shortcut).

    Raises:
        ImportError: If ``qdrant-client`` is not installed.
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        *,
        location: str = ":memory:",
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        """Initialize QdrantVectorStore."""
        _require_qdrant()
        if config is None:
            config = VectorStoreConfig(
                dimensions=dimensions or 384,
            )
        super().__init__(config)
        self._location = location
        self._api_key = api_key
        self._client: QdrantClient | None = None
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the Qdrant client and collection."""
        _require_qdrant()

        if self._location == ":memory:":
            self._client = QdrantClient(location=":memory:")
        elif self._location.startswith("http"):
            self._client = QdrantClient(
                url=self._location,
                api_key=self._api_key,
            )
        else:
            self._client = QdrantClient(path=self._location)

        distance = _DISTANCE_MAP.get(self.config.metric, Distance.COSINE)

        # Create collection if it doesn't exist
        collections = [c.name for c in self._client.get_collections().collections]
        if self.config.collection_name not in collections:
            self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.dimensions,
                    distance=distance,
                ),
            )
            logger.info(
                "Created Qdrant collection %r (dims=%d, distance=%s)",
                self.config.collection_name,
                self.config.dimensions,
                distance,
            )
        else:
            logger.info(
                "Using existing Qdrant collection %r",
                self.config.collection_name,
            )

        self._initialized = True

    def close(self) -> None:
        """Close the Qdrant client connection."""
        if self._client is not None:
            self._client.close()
        self._client = None
        self._initialized = False

    def _check_initialized(self) -> None:
        if not self._initialized or self._client is None:
            raise RuntimeError("Store not initialized. Call initialize() or use a context manager.")

    # ------------------------------------------------------------------
    # Payload helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_payload(item: EmbeddedChunk) -> dict[str, Any]:
        """Build a Qdrant payload from an EmbeddedChunk."""
        payload: dict[str, Any] = {
            "chunk_id": item.chunk.id,
            "document_id": item.chunk.document_id,
            "content": item.chunk.content,
            "chunk_index": item.chunk.index,
            "start_char": item.chunk.start_char,
            "end_char": item.chunk.end_char,
            "strategy": item.chunk.strategy.value,
            "embedding_model": item.embedding.model,
        }
        # Store user metadata as nested dict
        if item.chunk.metadata:
            payload["metadata"] = item.chunk.metadata
        return payload

    @staticmethod
    def _payload_to_embedded_chunk(
        point_id: str,
        payload: dict[str, Any],
        vector: list[float] | None = None,
    ) -> EmbeddedChunk:
        """Reconstruct an EmbeddedChunk from a Qdrant point."""
        chunk_id = payload.get("chunk_id", point_id)
        user_meta = payload.get("metadata", {})

        chunk = Chunk(
            content=payload.get("content", ""),
            document_id=payload.get("document_id", ""),
            index=payload.get("chunk_index", 0),
            start_char=payload.get("start_char", 0),
            end_char=payload.get("end_char", 0),
            metadata=user_meta if isinstance(user_meta, dict) else {},
            strategy=ChunkStrategy(payload.get("strategy", "fixed_size")),
            id=chunk_id,
        )
        embedding = Embedding(
            chunk_id=chunk_id,
            vector=vector or [],
            model=payload.get("embedding_model", ""),
        )
        return EmbeddedChunk(chunk=chunk, embedding=embedding)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, items: list[EmbeddedChunk]) -> list[str]:
        """Add embedded chunks as Qdrant points.

        Uses the chunk ID as the Qdrant point ID (UUID format).

        Args:
            items: Chunks with embeddings.

        Returns:
            List of chunk IDs.
        """
        self._check_initialized()
        assert self._client is not None
        if not items:
            return []

        points: list[PointStruct] = []
        ids: list[str] = []

        for item in items:
            cid = item.chunk.id
            # Qdrant needs UUID or integer point IDs
            try:
                point_id = str(uuid.UUID(cid))
            except ValueError:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, cid))

            points.append(
                PointStruct(
                    id=point_id,
                    vector=item.embedding.vector,
                    payload=self._build_payload(item),
                )
            )
            ids.append(cid)

        self._client.upsert(
            collection_name=self.config.collection_name,
            points=points,
        )
        logger.debug("Added %d points to Qdrant", len(ids))
        return ids

    def upsert(self, items: list[EmbeddedChunk]) -> list[str]:
        """Insert or update chunks in Qdrant.

        Qdrant's upsert is natively idempotent.

        Args:
            items: Chunks with embeddings.

        Returns:
            List of upserted IDs.
        """
        # Qdrant upsert = add (it's inherently an upsert)
        return self.add(items)

    def delete(self, ids: list[str]) -> int:
        """Remove points from Qdrant by chunk ID.

        Args:
            ids: Chunk IDs to delete.

        Returns:
            Number of IDs submitted for deletion.
        """
        self._check_initialized()
        assert self._client is not None
        if not ids:
            return 0

        # Use filter-based deletion on chunk_id payload field
        for cid in ids:
            self._client.delete(
                collection_name=self.config.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_id",
                            match=MatchValue(value=cid),
                        )
                    ]
                ),
            )
        return len(ids)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, id: str) -> EmbeddedChunk | None:
        """Retrieve a chunk by ID using payload filtering.

        Args:
            id: The chunk ID.

        Returns:
            The stored chunk or ``None``.
        """
        self._check_initialized()
        assert self._client is not None

        results = self._client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="chunk_id",
                        match=MatchValue(value=id),
                    )
                ]
            ),
            limit=1,
            with_vectors=True,
            with_payload=True,
        )

        points = results[0] if results else []
        if not points:
            return None

        point = points[0]
        vector = point.vector if isinstance(point.vector, list) else []
        return self._payload_to_embedded_chunk(str(point.id), point.payload or {}, vector)

    def list_ids(self) -> list[str]:
        """Return all stored chunk IDs.

        Returns:
            Sorted list of chunk IDs.
        """
        self._check_initialized()
        assert self._client is not None
        ids: list[str] = []
        offset = None

        while True:
            results, next_offset = self._client.scroll(
                collection_name=self.config.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                cid = (point.payload or {}).get("chunk_id", str(point.id))
                ids.append(cid)

            if next_offset is None:
                break
            offset = next_offset

        return sorted(ids)

    def count(self) -> int:
        """Return the number of points in the collection.

        Returns:
            Point count.
        """
        if not self._initialized or self._client is None:
            return 0
        info = self._client.get_collection(self.config.collection_name)
        return info.points_count or 0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _build_qdrant_filter(self, filters: dict[str, Any]) -> Any:
        """Convert LexiSearch metadata filters to a Qdrant Filter."""
        conditions: list[FieldCondition] = []
        for key, value in filters.items():
            if key == "document_id":
                conditions.append(FieldCondition(key="document_id", match=MatchValue(value=value)))
            else:
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value),
                    )
                )
        return Filter(must=conditions) if conditions else None

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Similarity search using Qdrant's HNSW index.

        Args:
            query_vector: Query embedding.
            top_k: Maximum results.
            filters: Metadata filters (converted to Qdrant Filter).

        Returns:
            Ordered search results.
        """
        self._check_initialized()
        assert self._client is not None
        if self.count() == 0:
            return []

        query_filter = self._build_qdrant_filter(filters) if filters else None

        hits = self._client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        results: list[SearchResult] = []
        for rank, hit in enumerate(hits, start=1):
            ec = self._payload_to_embedded_chunk(str(hit.id), hit.payload or {})
            results.append(SearchResult(chunk=ec.chunk, score=hit.score, rank=rank))

        return results

    def search_by_text(
        self,
        query: str,
        embedder: Any,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed a query and search Qdrant.

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
        """Create a Qdrant snapshot.

        For disk-backed and remote instances, data is persisted
        automatically.  This method triggers an explicit snapshot.

        Args:
            path: Ignored; Qdrant manages snapshot paths internally.
        """
        self._check_initialized()
        assert self._client is not None
        if self._location != ":memory:":
            self._client.create_snapshot(
                collection_name=self.config.collection_name,
            )
            logger.info("Qdrant snapshot created for %s", self.config.collection_name)
        else:
            logger.warning("Cannot persist in-memory Qdrant store")

    def load(self, path: str) -> None:
        """Re-initialise from a Qdrant disk path or remote URL.

        Args:
            path: Qdrant storage path or server URL.
        """
        self._location = path
        self.initialize()
