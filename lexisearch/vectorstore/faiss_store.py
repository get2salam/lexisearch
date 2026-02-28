"""FAISS-backed vector store for high-performance similarity search.

`FAISS <https://github.com/facebookresearch/faiss>`_ (Facebook AI Similarity
Search) provides efficient billion-scale vector search.  This module wraps
FAISS indexes behind the :class:`BaseVectorStore` interface.

Supported index types:

- **Flat** (``IndexFlatL2`` / ``IndexFlatIP``): exact search, no training.
- **IVF** (``IndexIVFFlat``): approximate search with inverted-file index.

Install FAISS via ``pip install faiss-cpu`` (or ``faiss-gpu`` for GPU).

Example:
    >>> from lexisearch.vectorstore.faiss_store import FAISSVectorStore
    >>> store = FAISSVectorStore(dimensions=384)
    >>> store.initialize()
"""

from __future__ import annotations

import logging
import pickle
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

logger = logging.getLogger(__name__)

try:
    import faiss
    import numpy as np

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def _require_faiss() -> None:
    """Raise a clear error when FAISS is not installed."""
    if not HAS_FAISS:
        raise ImportError(
            "FAISS is required for FAISSVectorStore. "
            "Install with: pip install faiss-cpu  (or faiss-gpu)"
        )


class FAISSVectorStore(BaseVectorStore):
    """FAISS-backed vector store.

    Args:
        config: Store configuration.
        index_type: ``"flat"`` for exact search or ``"ivf"`` for approximate.
        nlist: Number of IVF cells (only for ``index_type="ivf"``).
        nprobe: Number of cells to visit at query time (IVF only).

    Raises:
        ImportError: If ``faiss`` is not installed.
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        *,
        dimensions: int | None = None,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
    ) -> None:
        """Initialize FAISSVectorStore."""
        _require_faiss()
        if config is None:
            config = VectorStoreConfig(
                dimensions=dimensions or 384,
            )
        super().__init__(config)
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe

        self._index: Any | None = None
        self._id_map: dict[int, str] = {}  # FAISS int id → chunk str id
        self._reverse_map: dict[str, int] = {}  # chunk str id → FAISS int id
        self._payloads: dict[str, dict[str, Any]] = {}  # chunk id → serialised chunk data
        self._next_id: int = 0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the FAISS index."""
        _require_faiss()
        dims = self.config.dimensions

        if self.config.metric in (DistanceMetric.COSINE, DistanceMetric.DOT_PRODUCT):
            base_index = faiss.IndexFlatIP(dims)
        else:
            base_index = faiss.IndexFlatL2(dims)

        if self.index_type == "ivf":
            quantizer = base_index
            self._index = faiss.IndexIVFFlat(
                quantizer,
                dims,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT
                if self.config.metric != DistanceMetric.EUCLIDEAN
                else faiss.METRIC_L2,
            )
            self._index.nprobe = self.nprobe
        else:
            self._index = base_index

        self._id_map = {}
        self._reverse_map = {}
        self._payloads = {}
        self._next_id = 0
        self._initialized = True
        logger.info(
            "FAISSVectorStore initialized (type=%s, dims=%d, metric=%s)",
            self.index_type,
            dims,
            self.config.metric.value,
        )

    def close(self) -> None:
        """Release the FAISS index."""
        self._index = None
        self._initialized = False

    def _check_initialized(self) -> None:
        if not self._initialized or self._index is None:
            raise RuntimeError("Store not initialized. Call initialize() or use a context manager.")

    # ------------------------------------------------------------------
    # Vector preparation
    # ------------------------------------------------------------------

    def _prepare_vectors(self, vectors: list[list[float]]) -> Any:
        """Convert Python lists to a FAISS-compatible numpy array.

        For cosine similarity, vectors are L2-normalised before indexing.
        """
        arr = np.array(vectors, dtype=np.float32)
        if self.config.metric == DistanceMetric.COSINE:
            faiss.normalize_L2(arr)
        return arr

    # ------------------------------------------------------------------
    # Serialise / deserialise payloads
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_to_payload(item: EmbeddedChunk) -> dict[str, Any]:
        """Serialise an EmbeddedChunk's metadata for storage."""
        return {
            "chunk_id": item.chunk.id,
            "document_id": item.chunk.document_id,
            "content": item.chunk.content,
            "metadata": item.chunk.metadata,
            "chunk_index": item.chunk.index,
            "start_char": item.chunk.start_char,
            "end_char": item.chunk.end_char,
            "strategy": item.chunk.strategy.value,
            "embedding_model": item.embedding.model,
            "vector": item.embedding.vector,
        }

    @staticmethod
    def _payload_to_embedded_chunk(payload: dict[str, Any]) -> EmbeddedChunk:
        """Reconstruct an EmbeddedChunk from a stored payload."""
        chunk = Chunk(
            content=payload["content"],
            document_id=payload["document_id"],
            index=payload.get("chunk_index", 0),
            start_char=payload.get("start_char", 0),
            end_char=payload.get("end_char", 0),
            metadata=payload.get("metadata", {}),
            strategy=ChunkStrategy(payload.get("strategy", "fixed_size")),
            id=payload["chunk_id"],
        )
        embedding = Embedding(
            chunk_id=payload["chunk_id"],
            vector=payload.get("vector", []),
            model=payload.get("embedding_model", ""),
        )
        return EmbeddedChunk(chunk=chunk, embedding=embedding)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, items: list[EmbeddedChunk]) -> list[str]:
        """Add embedded chunks to the FAISS index.

        Args:
            items: Chunks with embeddings.

        Returns:
            List of chunk IDs.

        Raises:
            ValueError: On duplicate IDs or dimension mismatch.
        """
        self._check_initialized()
        assert self._index is not None
        ids: list[str] = []
        vectors: list[list[float]] = []

        for item in items:
            cid = item.chunk.id
            if cid in self._reverse_map:
                raise ValueError(f"Duplicate chunk ID: {cid}")
            if len(item.embedding.vector) != self.config.dimensions:
                raise ValueError(
                    f"Expected {self.config.dimensions}-dim vector, "
                    f"got {len(item.embedding.vector)}"
                )
            int_id = self._next_id
            self._next_id += 1
            self._id_map[int_id] = cid
            self._reverse_map[cid] = int_id
            self._payloads[cid] = self._chunk_to_payload(item)
            vectors.append(item.embedding.vector)
            ids.append(cid)

        if vectors:
            arr = self._prepare_vectors(vectors)
            self._index.add(arr)

        logger.debug("Added %d vectors to FAISS (total=%d)", len(ids), self._index.ntotal)
        return ids

    def upsert(self, items: list[EmbeddedChunk]) -> list[str]:
        """Insert or update chunks.

        For FAISS (which doesn't natively support updates), existing entries
        are deleted and re-added.  This requires an ``IndexIDMap`` wrapper
        for production use; here we rebuild the index for simplicity.

        Args:
            items: Chunks with embeddings.

        Returns:
            List of upserted IDs.
        """
        self._check_initialized()
        # Separate existing and new
        to_delete = [item.chunk.id for item in items if item.chunk.id in self._reverse_map]
        if to_delete:
            self.delete(to_delete)
        return self.add(items)

    def delete(self, ids: list[str]) -> int:
        """Remove items from the store.

        Since FAISS flat indexes don't support removal, we rebuild the
        index without the deleted items.

        Args:
            ids: Chunk IDs to delete.

        Returns:
            Number of items actually removed.
        """
        self._check_initialized()
        to_remove = set(ids) & set(self._reverse_map.keys())
        if not to_remove:
            return 0

        # Remove from maps
        for cid in to_remove:
            int_id = self._reverse_map.pop(cid)
            del self._id_map[int_id]
            del self._payloads[cid]

        # Rebuild index with remaining vectors
        self._rebuild_index()
        return len(to_remove)

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from payloads."""
        old_metric = self.config.metric
        dims = self.config.dimensions

        if old_metric in (DistanceMetric.COSINE, DistanceMetric.DOT_PRODUCT):
            self._index = faiss.IndexFlatIP(dims)
        else:
            self._index = faiss.IndexFlatL2(dims)

        # Re-add all remaining vectors
        new_id_map: dict[int, str] = {}
        new_reverse_map: dict[str, int] = {}
        vectors: list[list[float]] = []
        idx = 0

        for cid, payload in self._payloads.items():
            new_id_map[idx] = cid
            new_reverse_map[cid] = idx
            vectors.append(payload["vector"])
            idx += 1

        self._id_map = new_id_map
        self._reverse_map = new_reverse_map
        self._next_id = idx

        if vectors:
            arr = self._prepare_vectors(vectors)
            self._index.add(arr)

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
        payload = self._payloads.get(id)
        if payload is None:
            return None
        return self._payload_to_embedded_chunk(payload)

    def list_ids(self) -> list[str]:
        """Return all stored IDs in sorted order.

        Returns:
            Sorted chunk IDs.
        """
        self._check_initialized()
        return sorted(self._payloads.keys())

    def count(self) -> int:
        """Return the number of indexed vectors.

        Returns:
            Item count.
        """
        if not self._initialized or self._index is None:
            return 0
        return int(self._index.ntotal)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Similarity search using the FAISS index.

        Args:
            query_vector: Query embedding.
            top_k: Maximum results.
            filters: Metadata filters (post-search filtering).

        Returns:
            Ordered search results.
        """
        self._check_initialized()
        assert self._index is not None
        if self._index.ntotal == 0:
            return []

        if len(query_vector) != self.config.dimensions:
            raise ValueError(
                f"Expected {self.config.dimensions}-dim query, got {len(query_vector)}"
            )

        # Fetch more than top_k to allow for post-filtering
        fetch_k = min(top_k * 3, self._index.ntotal) if filters else min(top_k, self._index.ntotal)

        query_arr = self._prepare_vectors([query_vector])
        distances, indices = self._index.search(query_arr, fetch_k)

        results: list[SearchResult] = []
        rank = 1
        for i in range(fetch_k):
            idx = int(indices[0][i])
            if idx == -1:
                continue

            cid = self._id_map.get(idx)
            if cid is None:
                continue

            payload = self._payloads.get(cid)
            if payload is None:
                continue

            # Apply post-search metadata filters
            if filters:
                match = True
                for key, value in filters.items():
                    if key == "document_id":
                        if payload.get("document_id") != value:
                            match = False
                            break
                    elif (
                        key not in payload.get("metadata", {}) or payload["metadata"][key] != value
                    ):
                        match = False
                        break
                if not match:
                    continue

            score = float(distances[0][i])
            # Convert L2 distance to similarity score
            if self.config.metric == DistanceMetric.EUCLIDEAN:
                score = 1.0 / (1.0 + score)

            ec = self._payload_to_embedded_chunk(payload)
            results.append(SearchResult(chunk=ec.chunk, score=score, rank=rank))
            rank += 1

            if len(results) >= top_k:
                break

        return results

    def search_by_text(
        self,
        query: str,
        embedder: Any,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed a query and search.

        Args:
            query: Text query.
            embedder: Embedder with ``embed_text`` method.
            top_k: Maximum results.
            filters: Metadata filters.

        Returns:
            Ordered results.
        """
        query_vector = embedder.embed_text(query)
        return self.search(query_vector, top_k=top_k, filters=filters)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, path: str | None = None) -> None:
        """Save the FAISS index and metadata to disk.

        Creates two files:
        - ``<path>.index`` — the FAISS index binary.
        - ``<path>.meta`` — pickled metadata (id maps and payloads).

        Args:
            path: Base path (without extension).
        """
        self._check_initialized()
        assert self._index is not None
        base = Path(path or self.config.collection_name)
        base.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = str(base) + ".index"
        faiss.write_index(self._index, index_path)

        # Save metadata
        meta_path = str(base) + ".meta"
        meta = {
            "config": {
                "collection_name": self.config.collection_name,
                "dimensions": self.config.dimensions,
                "metric": self.config.metric.value,
            },
            "id_map": self._id_map,
            "reverse_map": self._reverse_map,
            "payloads": self._payloads,
            "next_id": self._next_id,
            "index_type": self.index_type,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        logger.info(
            "Persisted FAISS store to %s (.index + .meta), %d vectors",
            base,
            self._index.ntotal,
        )

    def load(self, path: str) -> None:
        """Load a FAISS index and metadata from disk.

        Args:
            path: Base path (without extension).

        Raises:
            FileNotFoundError: If files are missing.
        """
        _require_faiss()
        base = Path(path)
        index_path = str(base) + ".index"
        meta_path = str(base) + ".meta"

        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not Path(meta_path).exists():
            raise FileNotFoundError(f"FAISS metadata not found: {meta_path}")

        self._index = faiss.read_index(index_path)

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        cfg = meta["config"]
        self.config.collection_name = cfg["collection_name"]
        self.config.dimensions = cfg["dimensions"]
        self.config.metric = DistanceMetric(cfg["metric"])

        # Restore int keys from JSON (they may have been stringified)
        self._id_map = {int(k): v for k, v in meta["id_map"].items()}
        self._reverse_map = meta["reverse_map"]
        self._payloads = meta["payloads"]
        self._next_id = meta["next_id"]
        self.index_type = meta.get("index_type", "flat")
        self._initialized = True

        logger.info(
            "Loaded FAISS store from %s, %d vectors",
            base,
            self._index.ntotal,
        )
