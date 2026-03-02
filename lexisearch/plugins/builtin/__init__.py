"""Built-in LexiSearch plugins.

All plugins here are ready to use without additional dependencies.

Plugins:
    - :class:`~lexisearch.plugins.builtin.logging_plugin.LoggingPlugin` —
      Structured logging at every pipeline stage.
    - :class:`~lexisearch.plugins.builtin.metrics_plugin.MetricsPlugin` —
      Automatic LexiMetrics instrumentation.
    - :class:`~lexisearch.plugins.builtin.rate_limit_plugin.RateLimitPlugin` —
      Token-bucket rate limiter for query throughput control.
"""

from __future__ import annotations

from lexisearch.plugins.builtin.logging_plugin import LoggingPlugin
from lexisearch.plugins.builtin.metrics_plugin import MetricsPlugin
from lexisearch.plugins.builtin.rate_limit_plugin import (
    RateLimitError,
    RateLimitExceeded,
    RateLimitPlugin,
)

__all__ = [
    "LoggingPlugin",
    "MetricsPlugin",
    "RateLimitError",
    "RateLimitExceeded",
    "RateLimitPlugin",
]
