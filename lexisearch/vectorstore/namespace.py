"""Namespace (multi-tenant) wrapper for any BaseVectorStore.

A ``NamespacedVectorStore`` transparently prefixes every chunk ID with a
caller-supplied namespace string.  This lets multiple tenants share a single
vector index without their data leaking across partition boundaries.

Usage::

    from lexisearch.vectorstore import InMemoryVectorStore, VectorStoreConfig
    from lexisearch.vectorstore.namespace import NamespacedVectorStore

    config = VectorStoreConfig(dimensions=4)
    base = InMemoryVectorStore(config=config)
    base.initialize()

    tenant_a = NamespacedVectorStore(base, namespace="tenant-a")
    tenant_b = NamespacedVectorStore(base, namespace="tenant-b")

    tenant_a.add(chunks_a)   # stored as "tenant-a::chunk-id"
    tenant_b.add(chunks_b)   # stored as "tenant-b::chunk-id"

    # Each tenant only sees their own data
    results = tenant_a.search(query_embedding, top_k=5)

Design notes
------------
- The namespace is prepended using ``::`` as a separator (``<ns>::<id>``).
- The underlying store's ``search`` method is called with an oversampled
  ``top_k`` and results are post-filtered to the caller's namespace.
- ``delete``, ``get``, and ``count`` also apply the namespace prefix.
- The wrapper works with any store that implements the ``InMemoryVectorStore``
  interface (``add``, ``search``, ``delete``, ``get``, ``count``, ``list_ids``).
"""

from __future__ import annotations

import copy
from typing import Any

from lexisearch.models import Chunk, EmbeddedChunk, Embedding, SearchResult

_SEP = "::"


def _qualify(namespace: str, chunk_id: str) -> str:
    """Return ``<namespace>::<chunk_id>``."""
    if not namespace:
        return chunk_id
    return f"{namespace}{_SEP}{chunk_id}"


def _strip_ns(namespace: str, qualified_id: str) -> str:
    """Remove the namespace prefix from a qualified ID, if present."""
    prefix = f"{namespace}{_SEP}"
    if qualified_id.startswith(prefix):
        return qualified_id[len(prefix) :]
    return qualified_id


def _qualify_chunk(namespace: str, item: EmbeddedChunk) -> EmbeddedChunk:
    """Return a shallow copy of *item* with the chunk ID namespace-qualified."""
    if not namespace:
        return item

    new_chunk = Chunk(
        content=item.chunk.content,
        document_id=item.chunk.document_id,
        index=item.chunk.index,
        start_char=item.chunk.start_char,
        end_char=item.chunk.end_char,
        metadata=copy.copy(item.chunk.metadata),
        strategy=item.chunk.strategy,
        id=_qualify(namespace, item.chunk.id),
    )
    new_embedding = Embedding(
        chunk_id=new_chunk.id,
        vector=item.embedding.vector,
        model=item.embedding.model,
    )
    return EmbeddedChunk(chunk=new_chunk, embedding=new_embedding)


def _unqualify_result(namespace: str, result: SearchResult) -> SearchResult:
    """Return a shallow copy of *result* with the namespace prefix stripped."""
    if not namespace:
        return result

    new_chunk = Chunk(
        content=result.chunk.content,
        document_id=result.chunk.document_id,
        index=result.chunk.index,
        start_char=result.chunk.start_char,
        end_char=result.chunk.end_char,
        metadata=copy.copy(result.chunk.metadata),
        strategy=result.chunk.strategy,
        id=_strip_ns(namespace, result.chunk.id),
    )
    return SearchResult(chunk=new_chunk, score=result.score, rank=result.rank)


class NamespacedVectorStore:
    """Proxy around any vector store that scopes operations to a namespace.

    Parameters
    ----------
    store:
        The underlying vector store.  Must implement the ``InMemoryVectorStore``
        interface: ``add``, ``upsert``, ``search``, ``delete``, ``get``,
        ``count``, ``list_ids``.
    namespace:
        Tenant or collection identifier.  Must not contain ``"::"`` itself.
        Pass an empty string for the *root* namespace (no prefix applied).
    """

    def __init__(self, store: Any, namespace: str) -> None:
        """Initialise with an underlying store and a namespace string."""
        if _SEP in namespace:
            raise ValueError(
                f"Namespace must not contain the separator {_SEP!r}. Got: {namespace!r}"
            )
        self._store = store
        self.namespace = namespace

    # ------------------------------------------------------------------
    # Properties forwarded from the underlying store
    # ------------------------------------------------------------------

    @property
    def config(self) -> Any:
        """Return the underlying store's config object."""
        return self._store.config

    @property
    def dim(self) -> int:
        """Return the vector dimensionality."""
        return getattr(self._store, "dim", getattr(self.config, "dimensions", 0))

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, items: list[EmbeddedChunk]) -> list[str]:
        """Add *items* to the store, qualifying each chunk ID with the namespace."""
        qualified = [_qualify_chunk(self.namespace, item) for item in items]
        ids = self._store.add(qualified)
        # Return unqualified IDs to the caller
        return [_strip_ns(self.namespace, i) for i in ids]

    def upsert(self, items: list[EmbeddedChunk]) -> list[str]:
        """Upsert *items*, qualifying chunk IDs with the namespace."""
        qualified = [_qualify_chunk(self.namespace, item) for item in items]
        ids = self._store.upsert(qualified)
        return [_strip_ns(self.namespace, i) for i in ids]

    def delete(self, ids: list[str]) -> int:
        """Delete items by their *unqualified* IDs."""
        qualified_ids = [_qualify(self.namespace, i) for i in ids]
        return self._store.delete(qualified_ids)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, chunk_id: str) -> EmbeddedChunk | None:
        """Retrieve a chunk by its *unqualified* ID."""
        result = self._store.get(_qualify(self.namespace, chunk_id))
        if result is None:
            return None
        return _qualify_chunk("", result)  # already stripped by proxy

    def list_ids(self) -> list[str]:
        """Return all chunk IDs belonging to this namespace (unqualified)."""
        prefix = f"{self.namespace}{_SEP}" if self.namespace else ""
        all_ids: list[str] = self._store.list_ids()
        return [_strip_ns(self.namespace, i) for i in all_ids if i.startswith(prefix)]

    def count(self) -> int:
        """Return the number of chunks belonging to this namespace."""
        return len(self.list_ids())

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the store and return only results belonging to this namespace.

        Oversamples by 4x to account for cross-namespace results that will be
        filtered out before returning ``top_k`` hits.
        """
        oversample = max(top_k * 4, 20)

        try:
            raw: list[SearchResult] = self._store.search(
                query_vector, top_k=oversample, filters=filters
            )
        except TypeError:
            raw = self._store.search(query_vector, top_k=oversample)

        prefix = f"{self.namespace}{_SEP}" if self.namespace else ""
        matched: list[SearchResult] = []
        for result in raw:
            chunk_id = result.chunk.id
            if not prefix or chunk_id.startswith(prefix):
                matched.append(_unqualify_result(self.namespace, result))
                if len(matched) >= top_k:
                    break

        # Re-rank to ensure consistent rank values after filtering
        for rank, result in enumerate(matched, start=1):
            result.rank = rank

        return matched

    def search_by_text(
        self,
        query: str,
        embedder: Any,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed *query* and perform a namespace-scoped similarity search."""
        query_vector = embedder.embed_text(query)
        return self.search(query_vector, top_k=top_k, filters=filters)

    # ------------------------------------------------------------------
    # Clear helpers
    # ------------------------------------------------------------------

    def clear(self, *, all_namespaces: bool = False) -> None:
        """Clear data.

        Parameters
        ----------
        all_namespaces:
            If ``True``, clear the entire underlying store.  Defaults to
            ``False``, which removes only chunks belonging to this namespace.
        """
        if all_namespaces:
            fn = getattr(self._store, "clear", None)
            if fn is not None:
                fn()
            return

        # Selective clear
        ids_to_delete = [_qualify(self.namespace, i) for i in self.list_ids()]
        if ids_to_delete:
            self._store.delete(ids_to_delete)

    # ------------------------------------------------------------------
    # Context manager / repr
    # ------------------------------------------------------------------

    def __enter__(self) -> NamespacedVectorStore:
        """Support ``with`` statement."""
        return self

    def __exit__(self, *_: Any) -> None:
        """No-op exit."""

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"NamespacedVectorStore(namespace={self.namespace!r}, "
            f"store={type(self._store).__name__})"
        )
