"""Built-in metrics plugin — automatic LexiMetrics instrumentation."""

from __future__ import annotations

import time
from typing import Any

from lexisearch.observability.metrics import InMemoryMetricsCollector, LexiMetrics
from lexisearch.observability.middleware import get_metrics
from lexisearch.plugins.base import BasePlugin, PluginContext, PluginMeta


class MetricsPlugin(BasePlugin):
    """Automatically record :class:`LexiMetrics` for every pipeline hook.

    Uses the global :func:`~lexisearch.observability.middleware.get_metrics`
    collector by default, or a custom one passed at construction time.

    Args:
        collector: Optional custom metrics collector.  Defaults to the
            global :func:`get_metrics` instance.

    Example::

        from lexisearch.plugins.builtin.metrics_plugin import MetricsPlugin
        from lexisearch.plugins.registry import PluginRegistry

        registry = PluginRegistry()
        registry.register(MetricsPlugin(), auto_load=True)

        # After queries …
        from lexisearch.observability.middleware import get_metrics
        snapshot = get_metrics().snapshot()
    """

    def __init__(self, collector: InMemoryMetricsCollector | None = None) -> None:
        """Initialise the metrics plugin.

        Args:
            collector: Metrics collector to use.
        """
        super().__init__()
        self._collector = collector

    def _metrics(self) -> InMemoryMetricsCollector:
        return self._collector if self._collector is not None else get_metrics()

    @property
    def meta(self) -> PluginMeta:
        """Return metrics plugin metadata."""
        return PluginMeta(
            name="metrics",
            version="1.0.0",
            description="Automatic LexiMetrics instrumentation via plugin hooks",
            author="LexiSearch",
            tags=["builtin", "observability", "metrics"],
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def on_before_query(self, ctx: PluginContext) -> None:
        """Record query start time.

        Args:
            ctx: Plugin context.
        """
        ctx.get_plugin_data("metrics")["query_start"] = time.perf_counter()

    def on_after_query(self, ctx: PluginContext) -> None:
        """Record query counter and latency.

        Args:
            ctx: Plugin context.
        """
        m = self._metrics()
        pdata = ctx.get_plugin_data("metrics")
        m.increment(LexiMetrics.QUERIES_TOTAL)
        if "query_start" in pdata:
            elapsed_ms = (time.perf_counter() - pdata["query_start"]) * 1_000
            m.histogram(LexiMetrics.QUERY_LATENCY_MS, elapsed_ms)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def on_before_ingest(self, ctx: PluginContext) -> None:
        """Record ingest start time.

        Args:
            ctx: Plugin context.
        """
        ctx.get_plugin_data("metrics")["ingest_start"] = time.perf_counter()

    def on_after_ingest(self, ctx: PluginContext) -> None:
        """Record documents ingested and ingest latency.

        Args:
            ctx: Plugin context.
        """
        m = self._metrics()
        pdata = ctx.get_plugin_data("metrics")
        docs = ctx.data.get("documents", [])
        chunks = ctx.data.get("chunks", [])
        if hasattr(docs, "__len__"):
            m.increment(LexiMetrics.DOCUMENTS_INGESTED, float(len(docs)))
        if hasattr(chunks, "__len__"):
            m.increment(LexiMetrics.CHUNKS_CREATED, float(len(chunks)))
        if "ingest_start" in pdata:
            elapsed_ms = (time.perf_counter() - pdata["ingest_start"]) * 1_000
            m.histogram(LexiMetrics.INGEST_LATENCY_MS, elapsed_ms)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def on_before_retrieve(self, ctx: PluginContext) -> None:
        """Record retrieval start time.

        Args:
            ctx: Plugin context.
        """
        ctx.get_plugin_data("metrics")["retrieve_start"] = time.perf_counter()

    def on_after_retrieve(self, ctx: PluginContext) -> None:
        """Record retrieval latency.

        Args:
            ctx: Plugin context.
        """
        m = self._metrics()
        pdata = ctx.get_plugin_data("metrics")
        if "retrieve_start" in pdata:
            elapsed_ms = (time.perf_counter() - pdata["retrieve_start"]) * 1_000
            m.histogram(LexiMetrics.RETRIEVAL_LATENCY_MS, elapsed_ms)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def on_before_generate(self, ctx: PluginContext) -> None:
        """Record generation start time.

        Args:
            ctx: Plugin context.
        """
        ctx.get_plugin_data("metrics")["generate_start"] = time.perf_counter()

    def on_after_generate(self, ctx: PluginContext) -> None:
        """Record generation latency and token usage.

        Args:
            ctx: Plugin context.
        """
        m = self._metrics()
        pdata = ctx.get_plugin_data("metrics")
        if "generate_start" in pdata:
            elapsed_ms = (time.perf_counter() - pdata["generate_start"]) * 1_000
            m.histogram(LexiMetrics.GENERATION_LATENCY_MS, elapsed_ms)
        # Token accounting (optional — requires response to carry usage)
        response: Any = ctx.data.get("response")
        if response is not None and hasattr(response, "usage"):
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            if prompt_tokens:
                m.increment(LexiMetrics.TOKENS_PROMPT, float(prompt_tokens))
            if completion_tokens:
                m.increment(LexiMetrics.TOKENS_COMPLETION, float(completion_tokens))
            m.increment(LexiMetrics.TOKENS_TOTAL, float(prompt_tokens + completion_tokens))

    # ------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------

    def on_error(self, ctx: PluginContext, exc: Exception) -> None:
        """Increment the error counter.

        Args:
            ctx: Plugin context.
            exc: The exception that was raised.
        """
        self._metrics().increment(
            LexiMetrics.ERRORS_TOTAL,
            operation=ctx.operation,
            error_type=type(exc).__name__,
        )
