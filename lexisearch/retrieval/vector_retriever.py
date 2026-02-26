"""Vector retriever adapter for dense retrieval.

Wraps any :class:`~lexisearch.vectorstore.base.BaseVectorStore` as a
:class:`~lexisearch.retrieval.base.BaseRetriever`, enabling seamless
integration of vector stores into the retrieval pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lexisearch.retrieval.base import (
    BaseRetriever,
    MetadataFilter,
    RetrieverConfig,
    RetrieverType,
)

if TYPE_CHECKING:
    from lexisearch.embeddings.base import BaseEmbedder
    from lexisearch.models import SearchResult
    from lexisearch.vectorstore.base import BaseVectorStore


@dataclass
class VectorRetrieverConfig(RetrieverConfig):
    """Configuration for the vector retriever.

    Extends :class:`RetrieverConfig` with embedding-specific options.

    Attributes:
        normalize_scores: Whether to normalize similarity scores to [0, 1].
        include_embeddings: Whether to include embedding vectors in results.
    """

    normalize_scores: bool = False
    include_embeddings: bool = False


class VectorRetriever(BaseRetriever):
    """Dense retriever that delegates to a vector store backend.

    Embeds the query using the provided embedder, then performs similarity
    search against the vector store. Acts as the bridge between the
    retrieval interface and the vector storage layer.

    Args:
        vector_store: The vector store to search against.
        embedder: The embedding provider for query encoding.
        config: Retriever configuration.

    Example::

        from lexisearch.vectorstore import InMemoryVectorStore, VectorStoreConfig
        from lexisearch.embeddings import MockEmbedder

        store = InMemoryVectorStore(VectorStoreConfig(dimensions=384))
        store.initialize()
        embedder = MockEmbedder(dimensions=384)

        retriever = VectorRetriever(store, embedder)
        results = retriever.retrieve("machine learning", top_k=5)
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        config: VectorRetrieverConfig | None = None,
    ) -> None:
        """Initialize the instance."""
        super().__init__(config or VectorRetrieverConfig())
        self.vector_store = vector_store
        self.embedder = embedder
        self._config = config or VectorRetrieverConfig()

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string into a vector.

        Uses the embedder's caching layer for repeated queries.

        Args:
            query: Natural-language query.

        Returns:
            Query embedding vector.
        """
        return self.embedder.embed_text_cached(query)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: list[MetadataFilter] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Retrieve chunks using dense vector similarity search.

        Embeds the query and delegates to the vector store's search method.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results.
            filters: Metadata filters (converted to store-native format).
            **kwargs: Additional parameters passed to the vector store.

        Returns:
            Ordered list of :class:`SearchResult` (best first).
        """
        k = top_k or self.config.top_k
        query_vector = self.embed_query(query)

        # Convert MetadataFilter to dict format for vector store
        store_filters: dict[str, Any] | None = None
        if filters:
            store_filters = self._convert_filters(filters)

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=k,
            filters=store_filters,
        )

        # Annotate results with retriever metadata
        for i, result in enumerate(results):
            result.metadata["retriever"] = "vector"
            result.metadata["embedding_model"] = self.embedder.model_name()
            result.rank = i + 1

        # Normalize scores if configured
        if self._config.normalize_scores and results:
            results = self._normalize_scores(results)

        return results

    def retrieve_by_vector(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve chunks using a pre-computed query vector.

        Useful when the query has already been embedded or when using
        a different encoding for the query vector.

        Args:
            query_vector: Pre-computed query embedding.
            top_k: Maximum number of results.
            filters: Store-native metadata filters.

        Returns:
            Ordered list of :class:`SearchResult`.
        """
        k = top_k or self.config.top_k
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=k,
            filters=filters,
        )
        for i, result in enumerate(results):
            result.metadata["retriever"] = "vector"
            result.rank = i + 1
        return results

    def retriever_type(self) -> RetrieverType:
        """Return the retriever type.

        Returns:
            :attr:`RetrieverType.DENSE`
        """
        return RetrieverType.DENSE

    @staticmethod
    def _convert_filters(filters: list[MetadataFilter]) -> dict[str, Any]:
        """Convert :class:`MetadataFilter` objects to a dict for vector stores.

        Simple equality filters are passed as ``{field: value}``.

        Args:
            filters: Typed metadata filters.

        Returns:
            Dictionary suitable for vector store ``search()`` calls.
        """
        result: dict[str, Any] = {}
        for f in filters:
            # Simple equality mapping for basic store compatibility
            result[f.field] = f.value
        return result

    @staticmethod
    def _normalize_scores(results: list[SearchResult]) -> list[SearchResult]:
        """Normalize scores to the [0, 1] range using min-max scaling.

        Args:
            results: Search results with raw scores.

        Returns:
            Results with normalised scores.
        """
        if len(results) <= 1:
            if results:
                results[0].score = 1.0
            return results

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            for r in results:
                r.score = 1.0
        else:
            for r in results:
                r.score = (r.score - min_score) / score_range

        return results

    def get_config(self) -> dict[str, Any]:
        """Return the retriever configuration.

        Returns:
            Configuration dictionary including vector store and embedder info.
        """
        base = super().get_config()
        base.update(
            {
                "vector_store": type(self.vector_store).__name__,
                "embedder": self.embedder.model_name(),
                "dimensions": self.embedder.dimensions(),
                "normalize_scores": self._config.normalize_scores,
            }
        )
        return base
