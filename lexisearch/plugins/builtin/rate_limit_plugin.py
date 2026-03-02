"""Built-in rate-limiting plugin — token-bucket throttle for queries."""

from __future__ import annotations

import time
from typing import Any

from lexisearch.plugins.base import BasePlugin, PluginContext, PluginMeta


class RateLimitError(Exception):
    """Raised when the rate limit is exceeded and ``raise_on_exceed=True``."""


#: Backwards-compatible alias.
RateLimitExceeded = RateLimitError


class RateLimitPlugin(BasePlugin):
    """Token-bucket rate limiter applied before each query.

    Uses a standard token-bucket algorithm: tokens refill at *rate* per
    second up to *burst* capacity.  Each query consumes one token.

    Args:
        rate: Token refill rate (queries per second).
        burst: Maximum burst capacity (tokens).  Defaults to *rate*.
        raise_on_exceed: If ``True``, raise :class:`RateLimitExceeded`
            when the bucket is empty.  If ``False``, block until a token
            is available (default).

    Example::

        plugin = RateLimitPlugin(rate=10.0, burst=20)
        registry.register(plugin, auto_load=True)
    """

    def __init__(
        self,
        rate: float = 10.0,
        burst: float | None = None,
        raise_on_exceed: bool = False,
    ) -> None:
        """Initialise the rate-limit plugin.

        Args:
            rate: Tokens refilled per second.
            burst: Maximum bucket capacity.  Defaults to *rate*.
            raise_on_exceed: Raise exception instead of sleeping.
        """
        super().__init__()
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self._rate = rate
        self._burst = burst if burst is not None else rate
        self._raise = raise_on_exceed
        self._tokens: float = self._burst
        self._last_refill: float = time.monotonic()

    @property
    def meta(self) -> PluginMeta:
        """Return rate-limit plugin metadata."""
        return PluginMeta(
            name="rate_limit",
            version="1.0.0",
            description="Token-bucket rate limiter for query throughput control",
            author="LexiSearch",
            tags=["builtin", "safety", "rate-limiting"],
        )

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def _consume(self) -> None:
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return
        # Bucket empty
        wait = (1.0 - self._tokens) / self._rate
        if self._raise:
            raise RateLimitError(f"Rate limit exceeded ({self._rate} req/s). Retry in {wait:.2f}s.")
        time.sleep(wait)
        self._refill()
        self._tokens -= 1.0

    def on_before_query(self, ctx: PluginContext) -> None:
        """Consume one token before allowing the query to proceed.

        Args:
            ctx: Plugin context (unused directly).
        """
        self._consume()

    def on_load(self, config: dict[str, Any] | None = None) -> None:
        """Apply optional config overrides on load.

        Args:
            config: May contain ``rate``, ``burst``, and ``raise_on_exceed``.
        """
        if config:
            if "rate" in config:
                self._rate = float(config["rate"])
            if "burst" in config:
                self._burst = float(config["burst"])
            if "raise_on_exceed" in config:
                self._raise = bool(config["raise_on_exceed"])
            # Reset bucket
            self._tokens = self._burst
            self._last_refill = time.monotonic()

    @property
    def available_tokens(self) -> float:
        """Current token count (after refill calculation).

        Returns:
            Number of available tokens (0 to burst).
        """
        self._refill()
        return self._tokens
