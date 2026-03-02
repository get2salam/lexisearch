"""Base plugin interface and configuration models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Plugin lifecycle states
# ---------------------------------------------------------------------------


class PluginState(str, Enum):
    """Current lifecycle state of a plugin."""

    REGISTERED = "registered"
    LOADED = "loaded"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    UNLOADED = "unloaded"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------


@dataclass
class PluginMeta:
    """Static metadata describing a plugin.

    Attributes:
        name: Unique plugin identifier (snake_case).
        version: Semantic version string.
        description: Human-readable description.
        author: Plugin author name or email.
        tags: Arbitrary tag set for discovery/filtering.
        requires: Names of plugins this plugin depends on.
    """

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise metadata to a plain dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "tags": self.tags,
            "requires": self.requires,
        }


# ---------------------------------------------------------------------------
# Plugin context (runtime shared state)
# ---------------------------------------------------------------------------


@dataclass
class PluginContext:
    """Shared context passed to plugins during hook calls.

    Plugins should read from and write to this context to communicate
    with the pipeline and with other plugins.

    Attributes:
        operation: Current pipeline operation name.
        data: Mutable data bag — read/write freely.
        metadata: Immutable request metadata.
        plugin_data: Per-plugin isolated storage (keyed by plugin name).
    """

    operation: str
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    plugin_data: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_plugin_data(self, plugin_name: str) -> dict[str, Any]:
        """Return the isolated data dict for *plugin_name*.

        Args:
            plugin_name: The name of the plugin.

        Returns:
            The plugin's private data dict (created on first access).
        """
        if plugin_name not in self.plugin_data:
            self.plugin_data[plugin_name] = {}
        return self.plugin_data[plugin_name]


# ---------------------------------------------------------------------------
# Base plugin
# ---------------------------------------------------------------------------


class BasePlugin(ABC):
    """Abstract base class for all LexiSearch plugins.

    Subclass this and implement :meth:`meta` to register a plugin.
    Override hook methods to participate in the pipeline lifecycle.

    Lifecycle:
        1. :meth:`on_load` — called once when the plugin is loaded.
        2. :meth:`on_before_*` hooks — called before each operation.
        3. :meth:`on_after_*` hooks — called after each operation.
        4. :meth:`on_error` — called when the pipeline raises an exception.
        5. :meth:`on_unload` — called once when the plugin is removed.

    Example::

        class TimingPlugin(BasePlugin):
            @property
            def meta(self) -> PluginMeta:
                return PluginMeta(name="timing", description="Measure latency")

            def on_before_query(self, ctx: PluginContext) -> None:
                import time
                ctx.get_plugin_data("timing")["start"] = time.perf_counter()

            def on_after_query(self, ctx: PluginContext) -> None:
                import time
                start = ctx.get_plugin_data("timing").get("start", 0)
                elapsed = (time.perf_counter() - start) * 1000
                print(f"Query took {elapsed:.1f} ms")
    """

    def __init__(self) -> None:
        """Initialise the plugin."""
        self._state: PluginState = PluginState.REGISTERED

    @property
    @abstractmethod
    def meta(self) -> PluginMeta:
        """Return static metadata describing this plugin."""

    @property
    def state(self) -> PluginState:
        """Current lifecycle state."""
        return self._state

    # ------------------------------------------------------------------
    # Lifecycle hooks (override as needed)
    # ------------------------------------------------------------------

    def on_load(self, config: dict[str, Any] | None = None) -> None:  # noqa: B027
        """Called once when the plugin is loaded into the registry.

        Args:
            config: Optional configuration dict from the registry.
        """

    def on_unload(self) -> None:  # noqa: B027
        """Called once when the plugin is unloaded from the registry."""

    def on_error(self, ctx: PluginContext, exc: Exception) -> None:  # noqa: B027
        """Called when an unhandled exception occurs in the pipeline.

        Args:
            ctx: The current plugin context.
            exc: The exception that was raised.
        """

    # ------------------------------------------------------------------
    # Query hooks
    # ------------------------------------------------------------------

    def on_before_query(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called before a query is processed.

        Args:
            ctx: Plugin context with ``data["query"]`` set.
        """

    def on_after_query(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called after a query returns results.

        Args:
            ctx: Plugin context with ``data["results"]`` set.
        """

    # ------------------------------------------------------------------
    # Ingest hooks
    # ------------------------------------------------------------------

    def on_before_ingest(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called before documents are ingested.

        Args:
            ctx: Plugin context with ``data["documents"]`` set.
        """

    def on_after_ingest(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called after documents have been ingested.

        Args:
            ctx: Plugin context with ``data["chunks"]`` set.
        """

    # ------------------------------------------------------------------
    # Retrieval hooks
    # ------------------------------------------------------------------

    def on_before_retrieve(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called before retrieval is executed.

        Args:
            ctx: Plugin context with ``data["query"]`` and ``data["top_k"]``.
        """

    def on_after_retrieve(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called after retrieval returns candidates.

        Args:
            ctx: Plugin context with ``data["candidates"]`` set.
        """

    # ------------------------------------------------------------------
    # Generation hooks
    # ------------------------------------------------------------------

    def on_before_generate(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called before the LLM generates a response.

        Args:
            ctx: Plugin context with ``data["prompt"]`` set.
        """

    def on_after_generate(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called after the LLM returns a response.

        Args:
            ctx: Plugin context with ``data["response"]`` set.
        """

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.meta.name!r}, "
            f"version={self.meta.version!r}, "
            f"state={self._state.value!r})"
        )
