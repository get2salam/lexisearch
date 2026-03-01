"""Concrete cache backend implementations."""

from __future__ import annotations

import json
import logging
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from lexisearch.cache.base import BaseCache, CacheEntry, CacheStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory LRU cache
# ---------------------------------------------------------------------------


class InMemoryCache(BaseCache):
    """Thread-safe in-memory LRU cache with optional TTL.

    Uses an ``OrderedDict`` to implement O(1) LRU eviction.  All operations
    are protected by a ``threading.RLock`` for safe concurrent access.

    Parameters
    ----------
    max_size:
        Maximum number of entries to keep in memory.  When full, the least
        recently used entry is evicted.
    default_ttl:
        Default TTL in seconds for entries that don't specify their own.
        ``None`` means entries never expire by default.
    """

    def __init__(self, max_size: int = 1024, default_ttl: float | None = None) -> None:
        """Initialise the in-memory LRU cache."""
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ------------------------------------------------------------------
    # BaseCache implementation
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return the value for *key* or ``None`` on miss/expiry."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired:
                del self._store[key]
                self._evictions += 1
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        """Store *value* under *key*, evicting LRU if at capacity."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key].value = value
                self._store[key].ttl = effective_ttl
                return
            # Evict LRU if at capacity
            while len(self._store) >= self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                self._evictions += 1
                logger.debug("Evicted LRU key: %s", evicted_key)
            self._store[key] = CacheEntry(key=key, value=value, ttl=effective_ttl)

    def delete(self, key: str) -> bool:
        """Remove *key* from the cache.  Returns ``True`` if it existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> int:
        """Remove all entries and return the count removed."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                total_entries=len(self._store),
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of currently stored entries."""
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        """Return a human-readable summary of the cache state."""
        s = self.stats()
        return f"InMemoryCache(size={len(self)}/{self._max_size}, hit_rate={s.hit_rate:.1%})"


# ---------------------------------------------------------------------------
# Disk-based persistent cache
# ---------------------------------------------------------------------------


class DiskCache(BaseCache):
    """Persistent LRU cache stored as JSON files on disk.

    Each entry is stored as a separate ``<key>.json`` file inside
    ``cache_dir``.  An in-memory index tracks access order for LRU eviction.
    Expired files are removed lazily on access.

    Parameters
    ----------
    cache_dir:
        Directory where cache files are stored (created if absent).
    max_size:
        Maximum number of entries.  LRU eviction is applied on ``set``.
    default_ttl:
        Default TTL in seconds.  ``None`` = permanent.
    serialiser:
        ``"json"`` (default, human-readable) or ``"pickle"`` (supports
        arbitrary Python objects but not human-readable).
    """

    def __init__(
        self,
        cache_dir: str | Path = ".lexisearch_cache",
        max_size: int = 4096,
        default_ttl: float | None = None,
        serialiser: str = "json",
    ) -> None:
        """Initialise disk cache at *cache_dir*."""
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._serialiser = serialiser
        self._lock = threading.RLock()
        self._index: OrderedDict[str, float] = OrderedDict()  # key → created_at
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._load_index()

    def _load_index(self) -> None:
        """Rebuild the in-memory index from existing cache files."""
        files = sorted(self._dir.glob("*.cache"), key=lambda f: f.stat().st_mtime)
        for f in files:
            key = f.stem
            self._index[key] = f.stat().st_mtime

    def _key_path(self, key: str) -> Path:
        """Return the file path for *key* (safe filename via hex encoding)."""
        safe = key.replace(":", "_").replace("/", "_")[:64]
        return self._dir / f"{safe}.cache"

    def _write(self, path: Path, entry: CacheEntry) -> None:
        """Serialise *entry* to *path*."""
        data = {
            "key": entry.key,
            "value": entry.value,
            "created_at": entry.created_at,
            "accessed_at": entry.accessed_at,
            "ttl": entry.ttl,
            "hits": entry.hits,
        }
        if self._serialiser == "pickle":
            path.write_bytes(pickle.dumps(data))
        else:
            path.write_text(json.dumps(data, default=str), encoding="utf-8")

    def _read(self, path: Path) -> CacheEntry | None:
        """Deserialise an entry from *path*, returning ``None`` on error."""
        try:
            if self._serialiser == "pickle":
                data = pickle.loads(path.read_bytes())
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
            return CacheEntry(
                key=data["key"],
                value=data["value"],
                created_at=data.get("created_at", time.time()),
                accessed_at=data.get("accessed_at", time.time()),
                ttl=data.get("ttl"),
                hits=data.get("hits", 0),
            )
        except Exception:
            return None

    def get(self, key: str) -> Any | None:
        """Return the cached value or ``None`` on miss/expiry."""
        path = self._key_path(key)
        with self._lock:
            if not path.exists():
                self._misses += 1
                return None
            entry = self._read(path)
            if entry is None or entry.is_expired:
                path.unlink(missing_ok=True)
                self._index.pop(key, None)
                self._evictions += 1
                self._misses += 1
                return None
            self._index.move_to_end(key)
            entry.touch()
            self._write(path, entry)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        """Store *value* under *key* on disk."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        path = self._key_path(key)
        with self._lock:
            # Evict LRU if at capacity
            while len(self._index) >= self._max_size and self._index:
                lru_key, _ = self._index.popitem(last=False)
                self._key_path(lru_key).unlink(missing_ok=True)
                self._evictions += 1
            entry = CacheEntry(key=key, value=value, ttl=effective_ttl)
            self._write(path, entry)
            self._index[key] = entry.created_at
            self._index.move_to_end(key)

    def delete(self, key: str) -> bool:
        """Remove *key* from disk cache."""
        path = self._key_path(key)
        with self._lock:
            if path.exists():
                path.unlink()
                self._index.pop(key, None)
                return True
            return False

    def clear(self) -> int:
        """Delete all cache files."""
        with self._lock:
            count = 0
            for f in self._dir.glob("*.cache"):
                f.unlink()
                count += 1
            self._index.clear()
            return count

    def stats(self) -> CacheStats:
        """Return runtime statistics."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                total_entries=len(self._index),
            )

    def __len__(self) -> int:
        """Number of cached entries."""
        with self._lock:
            return len(self._index)

    def __repr__(self) -> str:
        """Return a human-readable summary of the disk cache state."""
        s = self.stats()
        return (
            f"DiskCache(dir={self._dir}, size={len(self)}/{self._max_size}, "
            f"hit_rate={s.hit_rate:.1%})"
        )


# ---------------------------------------------------------------------------
# Tiered (L1 + L2) cache
# ---------------------------------------------------------------------------


class TieredCache(BaseCache):
    """Two-tier cache (fast L1 + larger L2).

    Reads check L1 first; on miss, L2 is consulted and the result is
    promoted to L1.  Writes go to both tiers simultaneously.

    Parameters
    ----------
    l1:
        Fast, small cache (e.g. ``InMemoryCache``).
    l2:
        Larger, slower cache (e.g. ``DiskCache`` or another ``InMemoryCache``).
    promote_on_hit:
        If ``True`` (default), L2 hits are promoted to L1.
    """

    def __init__(
        self,
        l1: BaseCache,
        l2: BaseCache,
        *,
        promote_on_hit: bool = True,
    ) -> None:
        """Initialise tiered cache with L1 and L2 backends."""
        self._l1 = l1
        self._l2 = l2
        self._promote = promote_on_hit
        self._hits_l1 = 0
        self._hits_l2 = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Check L1 then L2; promote L2 hits to L1 if configured."""
        value = self._l1.get(key)
        if value is not None:
            self._hits_l1 += 1
            return value
        value = self._l2.get(key)
        if value is not None:
            self._hits_l2 += 1
            if self._promote:
                self._l1.set(key, value)
            return value
        self._misses += 1
        return None

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        """Write to both L1 and L2."""
        self._l1.set(key, value, ttl=ttl)
        self._l2.set(key, value, ttl=ttl)

    def delete(self, key: str) -> bool:
        """Remove from both tiers."""
        d1 = self._l1.delete(key)
        d2 = self._l2.delete(key)
        return d1 or d2

    def clear(self) -> int:
        """Clear both tiers and return total entries removed."""
        return self._l1.clear() + self._l2.clear()

    def stats(self) -> CacheStats:
        """Return aggregated statistics across both tiers."""
        s1 = self._l1.stats()
        s2 = self._l2.stats()
        return CacheStats(
            hits=self._hits_l1 + self._hits_l2,
            misses=self._misses,
            evictions=s1.evictions + s2.evictions,
            total_entries=s2.total_entries,  # L2 is the authoritative store
        )

    def __repr__(self) -> str:
        """Return a human-readable summary of the tiered cache."""
        return f"TieredCache(l1={self._l1!r}, l2={self._l2!r})"
