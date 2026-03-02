"""Built-in logging plugin — structured logs for every pipeline hook."""

from __future__ import annotations

import logging
import time

from lexisearch.plugins.base import BasePlugin, PluginContext, PluginMeta


class LoggingPlugin(BasePlugin):
    """Emit structured log messages at each pipeline stage.

    Logs entry/exit times, query text, result counts, and any errors.

    Args:
        logger_name: Python logger name.  Defaults to ``"lexisearch.pipeline"``.
        level: Log level for normal events (default ``logging.INFO``).
        error_level: Log level for errors (default ``logging.ERROR``).

    Example::

        from lexisearch.plugins.builtin.logging_plugin import LoggingPlugin
        from lexisearch.plugins.registry import PluginRegistry

        registry = PluginRegistry()
        registry.register(LoggingPlugin(), auto_load=True)
    """

    def __init__(
        self,
        logger_name: str = "lexisearch.pipeline",
        level: int = logging.INFO,
        error_level: int = logging.ERROR,
    ) -> None:
        """Initialise the logging plugin.

        Args:
            logger_name: Python logger name.
            level: Log level for normal events.
            error_level: Log level for error events.
        """
        super().__init__()
        self._logger = logging.getLogger(logger_name)
        self._level = level
        self._error_level = error_level

    @property
    def meta(self) -> PluginMeta:
        """Return logging plugin metadata."""
        return PluginMeta(
            name="logging",
            version="1.0.0",
            description="Structured logging for all pipeline hooks",
            author="LexiSearch",
            tags=["builtin", "observability", "logging"],
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def on_before_query(self, ctx: PluginContext) -> None:
        """Log the incoming query text.

        Args:
            ctx: Plugin context with optional ``data["query"]``.
        """
        query = ctx.data.get("query", "<unknown>")
        self._logger.log(self._level, "query.start", extra={"query": query})
        ctx.get_plugin_data("logging")["query_start"] = time.perf_counter()

    def on_after_query(self, ctx: PluginContext) -> None:
        """Log query completion and result count.

        Args:
            ctx: Plugin context with optional ``data["results"]``.
        """
        pdata = ctx.get_plugin_data("logging")
        elapsed_ms = (time.perf_counter() - pdata.get("query_start", time.perf_counter())) * 1_000
        results = ctx.data.get("results", [])
        self._logger.log(
            self._level,
            "query.done",
            extra={
                "result_count": len(results) if hasattr(results, "__len__") else "?",
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def on_before_ingest(self, ctx: PluginContext) -> None:
        """Log the start of an ingest operation.

        Args:
            ctx: Plugin context with optional ``data["documents"]``.
        """
        docs = ctx.data.get("documents", [])
        doc_count = len(docs) if hasattr(docs, "__len__") else "?"
        self._logger.log(
            self._level,
            "ingest.start",
            extra={"document_count": doc_count},
        )
        ctx.get_plugin_data("logging")["ingest_start"] = time.perf_counter()

    def on_after_ingest(self, ctx: PluginContext) -> None:
        """Log ingest completion and chunk count.

        Args:
            ctx: Plugin context with optional ``data["chunks"]``.
        """
        pdata = ctx.get_plugin_data("logging")
        elapsed_ms = (time.perf_counter() - pdata.get("ingest_start", time.perf_counter())) * 1_000
        chunks = ctx.data.get("chunks", [])
        self._logger.log(
            self._level,
            "ingest.done",
            extra={
                "chunk_count": len(chunks) if hasattr(chunks, "__len__") else "?",
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def on_before_retrieve(self, ctx: PluginContext) -> None:
        """Log the start of a retrieval operation.

        Args:
            ctx: Plugin context with optional ``data["query"]``.
        """
        self._logger.log(
            self._level,
            "retrieve.start",
            extra={"query": ctx.data.get("query", "<unknown>")},
        )
        ctx.get_plugin_data("logging")["retrieve_start"] = time.perf_counter()

    def on_after_retrieve(self, ctx: PluginContext) -> None:
        """Log retrieval completion and candidate count.

        Args:
            ctx: Plugin context with optional ``data["candidates"]``.
        """
        pdata = ctx.get_plugin_data("logging")
        elapsed_ms = (
            time.perf_counter() - pdata.get("retrieve_start", time.perf_counter())
        ) * 1_000
        candidates = ctx.data.get("candidates", [])
        self._logger.log(
            self._level,
            "retrieve.done",
            extra={
                "candidate_count": (len(candidates) if hasattr(candidates, "__len__") else "?"),
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def on_before_generate(self, ctx: PluginContext) -> None:
        """Log the start of a generation step.

        Args:
            ctx: Plugin context with optional ``data["prompt"]``.
        """
        prompt = ctx.data.get("prompt", "")
        prompt_preview = str(prompt)[:80] + "…" if len(str(prompt)) > 80 else str(prompt)
        self._logger.log(
            self._level,
            "generate.start",
            extra={"prompt_preview": prompt_preview},
        )
        ctx.get_plugin_data("logging")["generate_start"] = time.perf_counter()

    def on_after_generate(self, ctx: PluginContext) -> None:
        """Log generation completion.

        Args:
            ctx: Plugin context with optional ``data["response"]``.
        """
        pdata = ctx.get_plugin_data("logging")
        elapsed_ms = (
            time.perf_counter() - pdata.get("generate_start", time.perf_counter())
        ) * 1_000
        self._logger.log(
            self._level,
            "generate.done",
            extra={"elapsed_ms": round(elapsed_ms, 2)},
        )

    # ------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------

    def on_error(self, ctx: PluginContext, exc: Exception) -> None:
        """Log pipeline errors.

        Args:
            ctx: Plugin context at the time of the error.
            exc: The exception that was raised.
        """
        self._logger.log(
            self._error_level,
            "pipeline.error",
            extra={
                "operation": ctx.operation,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
