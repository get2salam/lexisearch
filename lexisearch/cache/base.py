"""Base cache interface and supporting types."""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------


def make_cache_key(*parts: Any, prefix: str = "") -> str:
    """Build a stable cache key from arbitrary arguments.

    Parameters
    ----------
    *parts:
        Key components (must be JSON-serialisable or convertible to string).
    prefix:
        Optional namespace prefix (e.g. ``"embed"``, ``"query"``).

    Returns:
    -------
    str
        A hex digest of the serialised key material.
    """
    raw = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"{prefix}:{digest}" if prefix else digest


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """A single cached value with metadata."""

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    ttl: float | None = None
    """Time-to-live in seconds.  ``None`` = never expires."""

    hits: int = 0
    """Number of times this entry has been served from cache."""

    @property
    def is_expired(self) -> bool:
        """Return ``True`` if the entry is past its TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def touch(self) -> None:
        """Update accessed_at and increment hit counter."""
        self.accessed_at = time.time()
        self.hits += 1


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Runtime statistics for a cache backend."""

    hits: int = 0
    """Number of successful cache lookups."""

    misses: int = 0
    """Number of cache misses."""

    evictions: int = 0
    """Number of entries evicted (LRU or TTL)."""

    total_entries: int = 0
    """Current number of entries stored."""

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were cache hits (0.0-1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        """Return a human-readable summary of cache statistics."""
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"evictions={self.evictions}, hit_rate={self.hit_rate:.1%}, "
            f"entries={self.total_entries})"
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseCache(ABC):
    """Abstract cache interface.

    All concrete backends must implement ``get``, ``set``, ``delete``,
    ``clear``, and ``stats``.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or ``None`` on a miss.

        Implementations must:
        - Return ``None`` for missing keys (do not raise).
        - Return ``None`` for expired entries and delete them.
        - Update hit/miss statistics.
        """

    @abstractmethod
    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        """Store *value* under *key*.

        Parameters
        ----------
        key:
            Cache key (use ``make_cache_key`` for composite keys).
        value:
            Arbitrary JSON-serialisable value.
        ttl:
            Optional time-to-live in seconds.  ``None`` = permanent.
        """

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove a key from the cache.

        Returns:
        -------
        bool
            ``True`` if the key existed and was deleted.
        """

    @abstractmethod
    def clear(self) -> int:
        """Remove all entries from the cache.

        Returns:
        -------
        int
            Number of entries that were removed.
        """

    @abstractmethod
    def stats(self) -> CacheStats:
        """Return runtime statistics for this cache."""

    def get_or_set(self, key: str, factory: Any, *, ttl: float | None = None) -> Any:
        """Return cached value or call *factory()* to compute and cache it.

        Parameters
        ----------
        key:
            Cache key.
        factory:
            Zero-argument callable that produces the value on a miss.
        ttl:
            Time-to-live for the newly cached entry.

        Returns:
        -------
        Any
            Cached or freshly computed value.
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        if value is not None:
            self.set(key, value, ttl=ttl)
        return value

    def __contains__(self, key: str) -> bool:
        """Support ``key in cache`` syntax."""
        return self.get(key) is not None
