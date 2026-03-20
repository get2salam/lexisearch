"""Tests for query result caching."""

from __future__ import annotations

import time

from lexisearch.models import Chunk, SearchResult
from lexisearch.retrieval.cache import (
    CacheEntry,
    CacheStats,
    LRUCache,
    TieredCache,
    TTLCache,
    _cache_key,
    _normalise_query,
)


def _make_results(n: int = 3) -> list[SearchResult]:
    return [
        SearchResult(
            chunk=Chunk(content=f"result {i}", document_id="doc"),
            score=1.0 - i * 0.1,
        )
        for i in range(n)
    ]


class TestNormaliseQuery:
    def test_lowercase(self) -> None:
        assert _normalise_query("Hello World") == "hello world"

    def test_collapse_whitespace(self) -> None:
        assert _normalise_query("hello   world") == "hello world"

    def test_strip(self) -> None:
        assert _normalise_query("  hello  ") == "hello"


class TestCacheKey:
    def test_deterministic(self) -> None:
        k1 = _cache_key("hello", 10)
        k2 = _cache_key("hello", 10)
        assert k1 == k2

    def test_different_queries(self) -> None:
        k1 = _cache_key("hello", 10)
        k2 = _cache_key("world", 10)
        assert k1 != k2

    def test_different_top_k(self) -> None:
        k1 = _cache_key("hello", 5)
        k2 = _cache_key("hello", 10)
        assert k1 != k2

    def test_case_insensitive(self) -> None:
        k1 = _cache_key("Hello World", 10)
        k2 = _cache_key("hello world", 10)
        assert k1 == k2


class TestCacheStats:
    def test_hit_rate(self) -> None:
        stats = CacheStats(hits=7, misses=3)
        assert abs(stats.hit_rate - 0.7) < 1e-6

    def test_hit_rate_zero(self) -> None:
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_total_requests(self) -> None:
        stats = CacheStats(hits=10, misses=5)
        assert stats.total_requests == 15


class TestCacheEntry:
    def test_age(self) -> None:
        entry = CacheEntry(key="test", value=[])
        time.sleep(0.05)
        assert entry.age_seconds >= 0.04


class TestLRUCache:
    def test_put_and_get(self) -> None:
        cache = LRUCache(max_size=10)
        results = _make_results()
        cache.put("test query", results)
        got = cache.get("test query")
        assert got is not None
        assert len(got) == 3

    def test_miss(self) -> None:
        cache = LRUCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_eviction(self) -> None:
        cache = LRUCache(max_size=2)
        cache.put("query1", _make_results())
        cache.put("query2", _make_results())
        cache.put("query3", _make_results())
        assert cache.get("query1") is None
        assert cache.get("query2") is not None
        assert cache.get("query3") is not None

    def test_lru_order(self) -> None:
        cache = LRUCache(max_size=2)
        cache.put("query1", _make_results())
        cache.put("query2", _make_results())
        cache.get("query1")  # Access query1, making query2 LRU
        cache.put("query3", _make_results())
        assert cache.get("query1") is not None
        assert cache.get("query2") is None

    def test_update_existing(self) -> None:
        cache = LRUCache(max_size=10)
        cache.put("query", _make_results(2))
        cache.put("query", _make_results(5))
        got = cache.get("query")
        assert got is not None
        assert len(got) == 5

    def test_stats(self) -> None:
        cache = LRUCache(max_size=10)
        cache.put("q1", _make_results())
        cache.get("q1")
        cache.get("q2")
        s = cache.stats()
        assert s.hits == 1
        assert s.misses == 1
        assert s.current_size == 1

    def test_clear(self) -> None:
        cache = LRUCache(max_size=10)
        cache.put("q1", _make_results())
        cache.clear()
        assert cache.get("q1") is None
        assert cache.stats().current_size == 0

    def test_case_insensitive_keys(self) -> None:
        cache = LRUCache(max_size=10)
        cache.put("Hello World", _make_results())
        assert cache.get("hello world") is not None


class TestTTLCache:
    def test_put_and_get(self) -> None:
        cache = TTLCache(max_size=10, ttl_seconds=60)
        cache.put("query", _make_results())
        assert cache.get("query") is not None

    def test_expiry(self) -> None:
        cache = TTLCache(max_size=10, ttl_seconds=0.05)
        cache.put("query", _make_results())
        time.sleep(0.1)
        assert cache.get("query") is None

    def test_not_expired(self) -> None:
        cache = TTLCache(max_size=10, ttl_seconds=10)
        cache.put("query", _make_results())
        assert cache.get("query") is not None

    def test_eviction_at_capacity(self) -> None:
        cache = TTLCache(max_size=2, ttl_seconds=60)
        cache.put("q1", _make_results())
        cache.put("q2", _make_results())
        cache.put("q3", _make_results())
        s = cache.stats()
        assert s.current_size <= 2

    def test_stats(self) -> None:
        cache = TTLCache(max_size=10, ttl_seconds=60)
        cache.put("q1", _make_results())
        cache.get("q1")
        cache.get("missing")
        s = cache.stats()
        assert s.hits == 1
        assert s.misses == 1

    def test_clear(self) -> None:
        cache = TTLCache(max_size=10, ttl_seconds=60)
        cache.put("q1", _make_results())
        cache.clear()
        assert cache.get("q1") is None


class TestTieredCache:
    def test_put_and_get(self) -> None:
        cache = TieredCache(l1_size=10, l2_size=100)
        cache.put("query", _make_results())
        assert cache.get("query") is not None

    def test_promotion(self) -> None:
        cache = TieredCache(l1_size=5, l2_size=50, promotion_threshold=2)
        cache.put("query", _make_results())
        cache.get("query")  # 1st access
        cache.get("query")  # 2nd access — should promote to L1
        l1_stats = cache._l1.stats()
        assert l1_stats.current_size >= 1

    def test_l2_fallback(self) -> None:
        cache = TieredCache(l1_size=1, l2_size=10)
        cache.put("q1", _make_results())
        cache.put("q2", _make_results())
        assert cache.get("q1") is not None
        assert cache.get("q2") is not None

    def test_stats(self) -> None:
        cache = TieredCache()
        cache.put("q1", _make_results())
        cache.get("q1")
        cache.get("missing")
        s = cache.stats()
        assert s.hits == 1
        assert s.misses == 1

    def test_clear(self) -> None:
        cache = TieredCache()
        cache.put("q1", _make_results())
        cache.clear()
        assert cache.get("q1") is None
        assert cache.stats().current_size == 0
