"""Tests for the LexiSearch caching layer."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lexisearch.cache import (
    DiskCache,
    InMemoryCache,
    TieredCache,
    cached_embedding,
    cached_llm,
    cached_retrieval,
)
from lexisearch.cache.base import CacheEntry, CacheStats, make_cache_key

# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    def test_returns_string(self) -> None:
        key = make_cache_key("foo", "bar")
        assert isinstance(key, str)

    def test_same_args_same_key(self) -> None:
        k1 = make_cache_key("hello", 42)
        k2 = make_cache_key("hello", 42)
        assert k1 == k2

    def test_different_args_different_key(self) -> None:
        k1 = make_cache_key("a", 1)
        k2 = make_cache_key("b", 2)
        assert k1 != k2

    def test_prefix_included_in_key(self) -> None:
        k1 = make_cache_key("foo", prefix="ns1")
        k2 = make_cache_key("foo", prefix="ns2")
        assert k1 != k2

    def test_key_without_prefix(self) -> None:
        key = make_cache_key("foo")
        assert ":" not in key  # no prefix separator

    def test_key_with_prefix(self) -> None:
        key = make_cache_key("foo", prefix="embed")
        assert key.startswith("embed:")

    def test_dict_args_stable(self) -> None:
        k1 = make_cache_key({"a": 1, "b": 2})
        k2 = make_cache_key({"b": 2, "a": 1})
        assert k1 == k2  # sort_keys=True

    def test_non_serialisable_falls_back_to_str(self) -> None:
        # Object with no JSON representation falls back to default=str
        class Foo:
            def __str__(self) -> str:
                return "Foo()"

        key = make_cache_key(Foo())
        assert isinstance(key, str)


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_not_expired_by_default(self) -> None:
        entry = CacheEntry(key="k", value="v")
        assert not entry.is_expired

    def test_expired_after_ttl(self) -> None:
        entry = CacheEntry(key="k", value="v", created_at=time.time() - 10, ttl=5.0)
        assert entry.is_expired

    def test_not_expired_within_ttl(self) -> None:
        entry = CacheEntry(key="k", value="v", ttl=3600.0)
        assert not entry.is_expired

    def test_no_ttl_never_expires(self) -> None:
        entry = CacheEntry(key="k", value="v", ttl=None)
        assert not entry.is_expired

    def test_touch_increments_hits(self) -> None:
        entry = CacheEntry(key="k", value="v")
        assert entry.hits == 0
        entry.touch()
        entry.touch()
        assert entry.hits == 2

    def test_touch_updates_accessed_at(self) -> None:
        entry = CacheEntry(key="k", value="v")
        old = entry.accessed_at
        time.sleep(0.01)
        entry.touch()
        assert entry.accessed_at >= old


# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------


class TestCacheStats:
    def test_hit_rate_zero_when_no_accesses(self) -> None:
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_one_hundred_percent(self) -> None:
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_fifty_percent(self) -> None:
        stats = CacheStats(hits=5, misses=5)
        assert stats.hit_rate == pytest.approx(0.5)

    def test_str_representation(self) -> None:
        stats = CacheStats(hits=3, misses=7)
        s = str(stats)
        assert "hits=3" in s
        assert "30.0%" in s


# ---------------------------------------------------------------------------
# InMemoryCache
# ---------------------------------------------------------------------------


class TestInMemoryCache:
    def setup_method(self) -> None:
        self.cache = InMemoryCache(max_size=10)

    def test_get_miss_returns_none(self) -> None:
        assert self.cache.get("nonexistent") is None

    def test_set_and_get(self) -> None:
        self.cache.set("key", "value")
        assert self.cache.get("key") == "value"

    def test_set_complex_value(self) -> None:
        value = {"answer": [1, 2, 3], "score": 0.99}
        self.cache.set("complex", value)
        assert self.cache.get("complex") == value

    def test_delete_existing_key(self) -> None:
        self.cache.set("k", "v")
        assert self.cache.delete("k") is True
        assert self.cache.get("k") is None

    def test_delete_nonexistent_key(self) -> None:
        assert self.cache.delete("no-such-key") is False

    def test_clear_returns_count(self) -> None:
        self.cache.set("a", 1)
        self.cache.set("b", 2)
        count = self.cache.clear()
        assert count == 2
        assert len(self.cache) == 0

    def test_lru_eviction(self) -> None:
        cache = InMemoryCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # Access "a" to make it most recently used
        cache.get("a")
        # Insert "d" — should evict "b" (LRU)
        cache.set("d", 4)
        assert cache.get("a") is not None
        assert cache.get("b") is None  # evicted
        assert cache.get("c") is not None
        assert cache.get("d") is not None

    def test_ttl_expiry(self) -> None:
        self.cache.set("temp", "value", ttl=0.05)
        assert self.cache.get("temp") == "value"
        time.sleep(0.1)
        assert self.cache.get("temp") is None

    def test_no_ttl_persists(self) -> None:
        self.cache.set("perm", "value")
        time.sleep(0.05)
        assert self.cache.get("perm") == "value"

    def test_stats_hits_misses(self) -> None:
        self.cache.set("x", 1)
        self.cache.get("x")  # hit
        self.cache.get("y")  # miss
        s = self.cache.stats()
        assert s.hits >= 1
        assert s.misses >= 1

    def test_stats_hit_rate(self) -> None:
        self.cache.set("x", 1)
        self.cache.get("x")
        self.cache.get("x")
        self.cache.get("missing")
        s = self.cache.stats()
        assert s.hit_rate > 0

    def test_contains_operator(self) -> None:
        self.cache.set("present", 42)
        assert "present" in self.cache
        assert "absent" not in self.cache

    def test_len(self) -> None:
        assert len(self.cache) == 0
        self.cache.set("a", 1)
        self.cache.set("b", 2)
        assert len(self.cache) == 2

    def test_overwrite_existing_key(self) -> None:
        self.cache.set("k", "old")
        self.cache.set("k", "new")
        assert self.cache.get("k") == "new"
        assert len(self.cache) == 1

    def test_default_ttl_applied(self) -> None:
        cache = InMemoryCache(max_size=10, default_ttl=0.05)
        cache.set("k", "v")
        time.sleep(0.1)
        assert cache.get("k") is None

    def test_get_or_set(self) -> None:
        result = self.cache.get_or_set("k", lambda: "computed")
        assert result == "computed"
        # Second call should hit cache, not call factory
        calls = [0]

        def factory() -> str:
            calls[0] += 1
            return "new"

        self.cache.get_or_set("k", factory)
        assert calls[0] == 0  # factory NOT called

    def test_repr_contains_size(self) -> None:
        r = repr(self.cache)
        assert "InMemoryCache" in r

    def test_thread_safety(self) -> None:
        """Concurrent set/get should not raise or corrupt data."""
        import threading

        cache = InMemoryCache(max_size=100)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for i in range(50):
                    cache.set(f"key-{i % 20}", i)
                    cache.get(f"key-{i % 20}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ---------------------------------------------------------------------------
# DiskCache
# ---------------------------------------------------------------------------


class TestDiskCache:
    def test_set_and_get(self, tmp_dir: Path) -> None:
        cache = DiskCache(tmp_dir)
        cache.set("k", {"data": [1, 2, 3]})
        assert cache.get("k") == {"data": [1, 2, 3]}

    def test_miss_returns_none(self, tmp_dir: Path) -> None:
        cache = DiskCache(tmp_dir)
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self, tmp_dir: Path) -> None:
        cache = DiskCache(tmp_dir)
        cache.set("temp", "value", ttl=0.05)
        time.sleep(0.1)
        assert cache.get("temp") is None

    def test_persistent_across_instances(self, tmp_dir: Path) -> None:
        cache1 = DiskCache(tmp_dir)
        cache1.set("k", "persisted")
        cache2 = DiskCache(tmp_dir)
        assert cache2.get("k") == "persisted"

    def test_delete(self, tmp_dir: Path) -> None:
        cache = DiskCache(tmp_dir)
        cache.set("k", "v")
        assert cache.delete("k") is True
        assert cache.get("k") is None

    def test_clear(self, tmp_dir: Path) -> None:
        cache = DiskCache(tmp_dir)
        cache.set("a", 1)
        cache.set("b", 2)
        count = cache.clear()
        assert count == 2
        assert len(cache) == 0

    def test_lru_eviction(self, tmp_dir: Path) -> None:
        cache = DiskCache(tmp_dir, max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # evicts LRU ("a")
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_stats(self, tmp_dir: Path) -> None:
        cache = DiskCache(tmp_dir)
        cache.set("k", "v")
        cache.get("k")
        cache.get("missing")
        s = cache.stats()
        assert s.hits >= 1
        assert s.misses >= 1


# ---------------------------------------------------------------------------
# TieredCache
# ---------------------------------------------------------------------------


class TestTieredCache:
    def _make_tiered(self) -> TieredCache:
        l1 = InMemoryCache(max_size=5)
        l2 = InMemoryCache(max_size=50)
        return TieredCache(l1, l2)

    def test_set_and_get(self) -> None:
        cache = self._make_tiered()
        cache.set("k", "v")
        assert cache.get("k") == "v"

    def test_miss_returns_none(self) -> None:
        cache = self._make_tiered()
        assert cache.get("absent") is None

    def test_l1_hit(self) -> None:
        l1 = InMemoryCache(max_size=10)
        l2 = InMemoryCache(max_size=50)
        cache = TieredCache(l1, l2)
        cache.set("k", "v")
        # Get from l1 (should be there)
        result = cache.get("k")
        assert result == "v"

    def test_l2_promotion_to_l1(self) -> None:
        l1 = InMemoryCache(max_size=10)
        l2 = InMemoryCache(max_size=50)
        cache = TieredCache(l1, l2, promote_on_hit=True)
        # Write only to L2
        l2.set("k", "from_l2")
        # Get — should promote to L1
        result = cache.get("k")
        assert result == "from_l2"
        assert l1.get("k") == "from_l2"

    def test_no_promotion_when_disabled(self) -> None:
        l1 = InMemoryCache(max_size=10)
        l2 = InMemoryCache(max_size=50)
        cache = TieredCache(l1, l2, promote_on_hit=False)
        l2.set("k", "v")
        cache.get("k")
        assert l1.get("k") is None  # not promoted

    def test_delete_removes_from_both(self) -> None:
        l1 = InMemoryCache(max_size=10)
        l2 = InMemoryCache(max_size=50)
        cache = TieredCache(l1, l2)
        cache.set("k", "v")
        cache.delete("k")
        assert l1.get("k") is None
        assert l2.get("k") is None

    def test_clear_both_tiers(self) -> None:
        l1 = InMemoryCache(max_size=10)
        l2 = InMemoryCache(max_size=50)
        cache = TieredCache(l1, l2)
        cache.set("a", 1)
        cache.set("b", 2)
        count = cache.clear()
        assert count == 4  # 2 in L1 + 2 in L2

    def test_stats_aggregated(self) -> None:
        cache = self._make_tiered()
        cache.set("x", 1)
        cache.get("x")  # hit
        cache.get("y")  # miss
        s = cache.stats()
        assert s.hits >= 1
        assert s.misses >= 1


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


class TestCachedRetrievalDecorator:
    def test_caches_result(self) -> None:
        cache = InMemoryCache()
        calls = [0]

        @cached_retrieval(cache)
        def search(query: str) -> list[str]:
            calls[0] += 1
            return [f"result for {query}"]

        r1 = search("test")
        r2 = search("test")
        assert r1 == r2
        assert calls[0] == 1  # only called once

    def test_different_args_different_cache(self) -> None:
        cache = InMemoryCache()

        @cached_retrieval(cache)
        def search(query: str) -> str:
            return f"result:{query}"

        r1 = search("a")
        r2 = search("b")
        assert r1 != r2

    def test_ttl_respected(self) -> None:
        cache = InMemoryCache()
        calls = [0]

        @cached_retrieval(cache, ttl=0.05)
        def fn(x: int) -> int:
            calls[0] += 1
            return x * 2

        fn(5)
        fn(5)
        assert calls[0] == 1
        time.sleep(0.1)
        fn(5)
        assert calls[0] == 2  # re-computed after TTL

    def test_none_result_not_cached(self) -> None:
        cache = InMemoryCache()
        calls = [0]

        @cached_retrieval(cache)
        def fn() -> None:
            calls[0] += 1
            return None

        fn()
        fn()
        assert calls[0] == 2  # None is not cached

    def test_cache_attached_to_function(self) -> None:
        cache = InMemoryCache()

        @cached_retrieval(cache)
        def fn() -> str:
            return "ok"

        assert fn._cache is cache  # type: ignore[attr-defined]

    def test_preserves_function_name(self) -> None:
        cache = InMemoryCache()

        @cached_retrieval(cache)
        def my_search_fn(q: str) -> str:
            return q

        assert my_search_fn.__name__ == "my_search_fn"


class TestCachedEmbeddingDecorator:
    def test_caches_embeddings(self) -> None:
        cache = InMemoryCache()
        calls = [0]

        @cached_embedding(cache)
        def embed(text: str) -> list[float]:
            calls[0] += 1
            return [0.1, 0.2, 0.3]

        e1 = embed("hello")
        e2 = embed("hello")
        assert e1 == e2
        assert calls[0] == 1

    def test_different_texts_different_embeddings(self) -> None:
        cache = InMemoryCache()

        @cached_embedding(cache)
        def embed(text: str) -> list[float]:
            return [float(ord(c)) for c in text[:3]]

        assert embed("abc") != embed("xyz")


class TestCachedLLMDecorator:
    def test_caches_llm_responses(self) -> None:
        cache = InMemoryCache()
        calls = [0]

        @cached_llm(cache)
        def generate(prompt: str) -> str:
            calls[0] += 1
            return f"Response to: {prompt}"

        r1 = generate("What is RAG?")
        r2 = generate("What is RAG?")
        assert r1 == r2
        assert calls[0] == 1

    def test_different_prompts_different_responses(self) -> None:
        cache = InMemoryCache()

        @cached_llm(cache)
        def generate(prompt: str) -> str:
            return f"Response: {prompt}"

        assert generate("A") != generate("B")
