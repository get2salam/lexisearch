"""Abstract base class for vector store backends.

Every vector store in LexiSearch implements this interface, enabling seamless
swapping between FAISS, ChromaDB, Qdrant, or a simple in-memory store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lexisearch.models import EmbeddedChunk, SearchResult


class DistanceMetric(Enum):
    """Supported distance / similarity metrics.

    Attributes:
        COSINE: Cosine similarity (1 - cosine distance).
        EUCLIDEAN: L2 (Euclidean) distance — smaller is more similar.
        DOT_PRODUCT: Inner product — larger is more similar.
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class VectorStoreConfig:
    """Shared configuration for vector store initialisation.

    Attributes:
        collection_name: Logical name for the index / collection.
        dimensions: Dimensionality of vectors stored.
        metric: Distance metric used for similarity search.
        extra: Backend-specific options passed through to the implementation.
    """

    collection_name: str = "default"
    dimensions: int = 384
    metric: DistanceMetric = DistanceMetric.COSINE
    extra: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract vector store with CRUD and search operations.

    Implementations must override every ``@abstractmethod``.  The base class
    provides some convenience wrappers (e.g. :meth:`upsert_one`).

    Args:
        config: A :class:`VectorStoreConfig` describing the store.
    """

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        """Initialize BaseVectorStore."""
        self.config = config or VectorStoreConfig()

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """Create or open the underlying index / collection.

        Must be called before any other operation.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the store."""
        ...

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @abstractmethod
    def add(self, items: list[EmbeddedChunk]) -> list[str]:
        """Insert embedded chunks into the store.

        Args:
            items: Chunks with their embeddings.

        Returns:
            List of IDs for the inserted items.
        """
        ...

    @abstractmethod
    def upsert(self, items: list[EmbeddedChunk]) -> list[str]:
        """Insert or update embedded chunks.

        If a chunk with the same ID already exists its vector and payload
        are replaced.

        Args:
            items: Chunks with their embeddings.

        Returns:
            List of IDs for the upserted items.
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """Remove items by their IDs.

        Args:
            ids: Chunk IDs to delete.

        Returns:
            Number of items actually deleted.
        """
        ...

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    def get(self, id: str) -> EmbeddedChunk | None:
        """Retrieve a single item by ID.

        Args:
            id: The chunk ID.

        Returns:
            The stored :class:`EmbeddedChunk` or ``None``.
        """
        ...

    @abstractmethod
    def list_ids(self) -> list[str]:
        """Return all stored chunk IDs.

        Returns:
            Sorted list of IDs.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of items in the store.

        Returns:
            Item count.
        """
        ...

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Find the *top_k* most similar items to *query_vector*.

        Args:
            query_vector: The query embedding.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters (backend-specific).

        Returns:
            Ordered list of :class:`SearchResult` (best first).
        """
        ...

    @abstractmethod
    def search_by_text(
        self,
        query: str,
        embedder: Any,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed a text query and perform similarity search.

        Convenience method that combines embedding + search in one call.

        Args:
            query: Natural-language query.
            embedder: An embedder instance with an ``embed_text`` method.
            top_k: Maximum results.
            filters: Metadata filters.

        Returns:
            Ordered list of :class:`SearchResult`.
        """
        ...

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @abstractmethod
    def persist(self, path: str | None = None) -> None:
        """Persist the store to disk.

        Args:
            path: Optional path override; otherwise use the store default.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously persisted store from disk.

        Args:
            path: Path to the persisted data.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def add_one(self, item: EmbeddedChunk) -> str:
        """Insert a single embedded chunk.

        Args:
            item: The chunk to insert.

        Returns:
            The ID of the inserted item.
        """
        ids = self.add([item])
        return ids[0]

    def upsert_one(self, item: EmbeddedChunk) -> str:
        """Insert or update a single embedded chunk.

        Args:
            item: The chunk to upsert.

        Returns:
            The ID of the upserted item.
        """
        ids = self.upsert([item])
        return ids[0]

    def delete_one(self, id: str) -> bool:
        """Delete a single item by ID.

        Args:
            id: Chunk ID to delete.

        Returns:
            ``True`` if the item was deleted, ``False`` otherwise.
        """
        return self.delete([id]) > 0

    def clear(self) -> int:
        """Remove **all** items from the store.

        Returns:
            Number of items removed.
        """
        all_ids = self.list_ids()
        if not all_ids:
            return 0
        return self.delete(all_ids)

    def get_config(self) -> dict[str, Any]:
        """Return the store configuration as a serialisable dictionary.

        Returns:
            Configuration dict.
        """
        return {
            "collection_name": self.config.collection_name,
            "dimensions": self.config.dimensions,
            "metric": self.config.metric.value,
            "extra": self.config.extra,
            "backend": type(self).__name__,
            "count": self.count(),
        }

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> BaseVectorStore:
        """Enter the context manager."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{type(self).__name__}("
            f"collection={self.config.collection_name!r}, "
            f"dims={self.config.dimensions}, "
            f"metric={self.config.metric.value!r}, "
            f"count={self.count()})"
        )
