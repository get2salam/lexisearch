"""LexiSearch caching layer.

Provides pluggable caching backends for query results, embeddings, and
LLM responses.  All caches share a common ``BaseCache`` interface so backends
can be swapped without touching application code.

Backends
--------
``InMemoryCache``
    Thread-safe LRU cache backed by ``functools.lru_cache``-style eviction.
    Zero dependencies, fast, ideal for single-process deployments.

``DiskCache``
    Persistent LRU cache stored as JSON/pickle files on disk.  Survives
    process restarts; good for development and batch pipelines.

``TieredCache``
    Composes an L1 (fast, small) and L2 (slower, large) cache.  Reads check
    L1 first; misses fall through to L2 and populate L1.

Quick start::

    from lexisearch.cache import InMemoryCache, cached_retrieval

    cache = InMemoryCache(max_size=256)

    # Manual use
    cache.set("my-key", {"answer": "42"})
    result = cache.get("my-key")

    # Decorator
    @cached_retrieval(cache)
    def my_retriever(query: str) -> list[str]:
        ...
"""

from __future__ import annotations

from lexisearch.cache.backends import DiskCache, InMemoryCache, TieredCache
from lexisearch.cache.base import BaseCache, CacheEntry, CacheStats
from lexisearch.cache.decorators import cached_embedding, cached_llm, cached_retrieval

__all__ = [
    "BaseCache",
    "CacheEntry",
    "CacheStats",
    "DiskCache",
    "InMemoryCache",
    "TieredCache",
    "cached_embedding",
    "cached_llm",
    "cached_retrieval",
]
