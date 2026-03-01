"""Cache decorator helpers for the LexiSearch pipeline components."""

from __future__ import annotations

import functools
import logging
from typing import Any

from lexisearch.cache.base import BaseCache, make_cache_key

logger = logging.getLogger(__name__)


def cached_retrieval(
    cache: BaseCache,
    *,
    ttl: float | None = None,
    prefix: str = "retrieval",
) -> Any:
    """Decorator that caches the return value of a retrieval function.

    The cache key is derived from all positional and keyword arguments passed
    to the decorated function.

    Parameters
    ----------
    cache:
        A ``BaseCache`` instance to use for storage.
    ttl:
        Optional TTL (seconds) for cached entries.
    prefix:
        Namespace prefix for cache keys.

    Returns:
    -------
    Callable
        A wrapper that returns cached values on repeated calls.

    Example:
    -------
    ::

        from lexisearch.cache import InMemoryCache, cached_retrieval

        cache = InMemoryCache(max_size=256)

        @cached_retrieval(cache, ttl=300)
        def search(query: str, top_k: int = 5) -> list[dict]:
            ...
    """

    def decorator(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = make_cache_key(fn.__name__, *args, **kwargs, prefix=prefix)
            cached = cache.get(key)
            if cached is not None:
                logger.debug("Cache HIT [%s]: %s", prefix, key[:24])
                return cached
            logger.debug("Cache MISS [%s]: %s", prefix, key[:24])
            result = fn(*args, **kwargs)
            if result is not None:
                cache.set(key, result, ttl=ttl)
            return result

        wrapper._cache = cache  # type: ignore[attr-defined]
        wrapper._cache_prefix = prefix  # type: ignore[attr-defined]
        return wrapper

    return decorator


def cached_embedding(
    cache: BaseCache,
    *,
    ttl: float | None = None,
    prefix: str = "embed",
) -> Any:
    """Decorator that caches embedding vectors.

    Embeddings are expensive to compute; this decorator ensures each unique
    text is embedded at most once per cache lifetime.

    Parameters
    ----------
    cache:
        A ``BaseCache`` instance (``InMemoryCache`` recommended for speed).
    ttl:
        Optional TTL.  ``None`` = cache forever (embeddings are deterministic).
    prefix:
        Namespace prefix for cache keys.

    Example:
    -------
    ::

        @cached_embedding(InMemoryCache(max_size=8192))
        def embed(text: str) -> list[float]:
            return model.encode(text).tolist()
    """

    def decorator(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = make_cache_key(fn.__name__, *args, **kwargs, prefix=prefix)
            cached = cache.get(key)
            if cached is not None:
                logger.debug("Embedding cache HIT: %s", key[:24])
                return cached
            result = fn(*args, **kwargs)
            if result is not None:
                cache.set(key, result, ttl=ttl)
            return result

        wrapper._cache = cache  # type: ignore[attr-defined]
        return wrapper

    return decorator


def cached_llm(
    cache: BaseCache,
    *,
    ttl: float | None = None,
    prefix: str = "llm",
) -> Any:
    """Decorator that caches LLM generation responses.

    Useful for deterministic prompts (temperature=0) where the same prompt
    always produces the same output.  For stochastic generation, use a
    short TTL or disable caching.

    Parameters
    ----------
    cache:
        A ``BaseCache`` instance.
    ttl:
        Optional TTL.  For stochastic LLMs, set a short TTL (e.g. 60s).
    prefix:
        Namespace prefix for cache keys.

    Example:
    -------
    ::

        @cached_llm(DiskCache(".cache/llm"), ttl=3600)
        def generate(prompt: str, **kwargs: Any) -> str:
            return llm_client.complete(prompt, **kwargs)
    """

    def decorator(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = make_cache_key(fn.__name__, *args, **kwargs, prefix=prefix)
            cached = cache.get(key)
            if cached is not None:
                logger.debug("LLM cache HIT: %s", key[:24])
                return cached
            result = fn(*args, **kwargs)
            if result is not None:
                cache.set(key, result, ttl=ttl)
            return result

        wrapper._cache = cache  # type: ignore[attr-defined]
        return wrapper

    return decorator


def invalidate_cache(fn: Any, key_parts: tuple[Any, ...]) -> bool:
    """Manually invalidate a specific cache entry for a decorated function.

    Parameters
    ----------
    fn:
        A function decorated with one of the ``cached_*`` decorators.
    key_parts:
        The arguments that were passed to *fn* when the entry was created.

    Returns:
    -------
    bool
        ``True`` if an entry was found and deleted.
    """
    cache: BaseCache | None = getattr(fn, "_cache", None)
    prefix: str = getattr(fn, "_cache_prefix", "")
    if cache is None:
        logger.warning("Function %r has no attached cache", fn)
        return False
    key = make_cache_key(fn.__name__, *key_parts, prefix=prefix)
    return cache.delete(key)
