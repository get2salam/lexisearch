"""Query result caching for retrieval pipelines.

Provides LRU and TTL-based caching to avoid redundant embedding
computations and database lookups for repeated or similar queries.

Cache strategies::

    BaseCache
    ├── LRUCache          (evicts least recently used)
    ├── TTLCache          (evicts after time-to-live expires)
    └── TieredCache       (L1 LRU + L2 TTL, automatic promotion)

Usage::

    from lexisearch.retrieval.cache import TTLCache

    cache = TTLCache(max_size=1000, ttl_seconds=300)
    cache.put("contract law damages", search_results)
    results = cache.get("contract law damages")  # Cache hit
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lexisearch.models import SearchResult


@dataclass
class CacheStats:
    """Cache performance statistics.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        evictions: Number of entries evicted.
        current_size: Current number of entries.
        max_size: Maximum capacity.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction (0.0 to 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        """Total cache lookups."""
        return self.hits + self.misses


@dataclass
class CacheEntry:
    """A single cache entry with metadata.

    Attributes:
        key: Cache key (query hash).
        value: Cached search results.
        created_at: Unix timestamp when entry was created.
        last_accessed: Unix timestamp of last access.
        access_count: Number of times accessed.
    """

    key: str
    value: list[SearchResult]
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1

    @property
    def age_seconds(self) -> float:
        """Seconds since entry was created."""
        return time.time() - self.created_at


def _normalise_query(query: str) -> str:
    """Normalise a query string for cache key generation."""
    return " ".join(query.lower().split())


def _cache_key(query: str, top_k: int = 10, filters: str = "") -> str:
    """Generate a deterministic cache key from query parameters."""
    normalised = _normalise_query(query)
    raw = f"{normalised}|k={top_k}|f={filters}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class BaseCache(ABC):
    """Abstract base for query result caches."""

    @abstractmethod
    def get(self, query: str, top_k: int = 10, filters: str = "") -> list[SearchResult] | None:
        """Retrieve cached results for a query.

        Args:
            query: Search query string.
            top_k: Number of results requested.
            filters: Serialised filter string.

        Returns:
            Cached results if hit, None if miss.
        """

    @abstractmethod
    def put(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
        filters: str = "",
    ) -> None:
        """Store results in cache.

        Args:
            query: Search query string.
            results: Results to cache.
            top_k: Number of results.
            filters: Serialised filter string.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from cache."""

    @abstractmethod
    def stats(self) -> CacheStats:
        """Return cache performance statistics."""


class LRUCache(BaseCache):
    """Least Recently Used cache for search results.

    Evicts the least recently accessed entry when capacity is reached.

    Args:
        max_size: Maximum number of entries to cache.
    """

    def __init__(self, max_size: int = 500) -> None:
        """Initialise with maximum cache size."""
        self._max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, query: str, top_k: int = 10, filters: str = "") -> list[SearchResult] | None:
        """Retrieve from cache, promoting to most-recently-used."""
        key = _cache_key(query, top_k, filters)
        if key in self._cache:
            self._hits += 1
            entry = self._cache[key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._cache.move_to_end(key)
            return entry.value
        self._misses += 1
        return None

    def put(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
        filters: str = "",
    ) -> None:
        """Store in cache, evicting LRU if at capacity."""
        key = _cache_key(query, top_k, filters)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key].value = results
            self._cache[key].last_accessed = time.time()
            return

        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
            self._evictions += 1

        self._cache[key] = CacheEntry(key=key, value=results)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def stats(self) -> CacheStats:
        """Return cache statistics."""
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            current_size=len(self._cache),
            max_size=self._max_size,
        )


class TTLCache(BaseCache):
    """Time-To-Live cache that expires entries after a duration.

    Args:
        max_size: Maximum number of entries.
        ttl_seconds: Time in seconds before an entry expires.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0) -> None:
        """Initialise with max size and TTL."""
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if an entry has expired."""
        return entry.age_seconds > self._ttl

    def _cleanup(self) -> None:
        """Remove expired entries."""
        expired = [k for k, v in self._cache.items() if self._is_expired(v)]
        for k in expired:
            del self._cache[k]
            self._evictions += 1

    def get(self, query: str, top_k: int = 10, filters: str = "") -> list[SearchResult] | None:
        """Retrieve from cache if not expired."""
        key = _cache_key(query, top_k, filters)
        if key in self._cache:
            entry = self._cache[key]
            if not self._is_expired(entry):
                self._hits += 1
                entry.last_accessed = time.time()
                entry.access_count += 1
                return entry.value
            del self._cache[key]
            self._evictions += 1

        self._misses += 1
        return None

    def put(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
        filters: str = "",
    ) -> None:
        """Store in cache with TTL."""
        self._cleanup()

        key = _cache_key(query, top_k, filters)
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest = min(self._cache, key=lambda k: self._cache[k].created_at)
            del self._cache[oldest]
            self._evictions += 1

        self._cache[key] = CacheEntry(key=key, value=results)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def stats(self) -> CacheStats:
        """Return cache statistics."""
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            current_size=len(self._cache),
            max_size=self._max_size,
        )


class TieredCache(BaseCache):
    """Two-tier cache: fast L1 (LRU) backed by larger L2 (TTL).

    Frequently accessed queries stay in the small, fast L1 cache.
    Less frequent queries live in the larger L2 with TTL expiry.

    Args:
        l1_size: Maximum L1 (hot) cache entries.
        l2_size: Maximum L2 (warm) cache entries.
        l2_ttl: TTL in seconds for L2 entries.
        promotion_threshold: Access count to promote from L2 to L1.
    """

    def __init__(
        self,
        l1_size: int = 100,
        l2_size: int = 1000,
        l2_ttl: float = 600.0,
        promotion_threshold: int = 3,
    ) -> None:
        """Initialise tiered cache."""
        self._l1 = LRUCache(max_size=l1_size)
        self._l2 = TTLCache(max_size=l2_size, ttl_seconds=l2_ttl)
        self._promotion_threshold = promotion_threshold
        self._hits = 0
        self._misses = 0

    def get(self, query: str, top_k: int = 10, filters: str = "") -> list[SearchResult] | None:
        """Check L1 first, then L2. Promote hot L2 entries to L1."""
        result = self._l1.get(query, top_k, filters)
        if result is not None:
            self._hits += 1
            return result

        result = self._l2.get(query, top_k, filters)
        if result is not None:
            self._hits += 1
            key = _cache_key(query, top_k, filters)
            if key in self._l2._cache:
                entry = self._l2._cache[key]
                if entry.access_count >= self._promotion_threshold:
                    self._l1.put(query, result, top_k, filters)
            return result

        self._misses += 1
        return None

    def put(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
        filters: str = "",
    ) -> None:
        """Store in L2 by default. Hot queries auto-promote to L1."""
        self._l2.put(query, results, top_k, filters)

    def clear(self) -> None:
        """Clear both tiers."""
        self._l1.clear()
        self._l2.clear()

    def stats(self) -> CacheStats:
        """Return combined statistics."""
        l1 = self._l1.stats()
        l2 = self._l2.stats()
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=l1.evictions + l2.evictions,
            current_size=l1.current_size + l2.current_size,
            max_size=l1.max_size + l2.max_size,
        )
