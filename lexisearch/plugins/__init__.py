"""LexiSearch plugin system — extensible hook-based pipeline extensions.

The plugin system provides a lifecycle-managed, hook-based extension point for
every stage of the RAG pipeline.  Plugins are registered in a
:class:`~lexisearch.plugins.registry.PluginRegistry` and fire hooks
automatically before and after each operation.

Quick start::

    from lexisearch.plugins import (
        BasePlugin,
        PluginContext,
        PluginMeta,
        PluginRegistry,
        register_plugin,
    )
    from lexisearch.plugins.builtin import LoggingPlugin, MetricsPlugin

    # Use the global registry
    register_plugin(LoggingPlugin())
    register_plugin(MetricsPlugin())

    # Or create an isolated registry
    registry = PluginRegistry()
    registry.register(LoggingPlugin(), auto_load=True)

    # Fire hooks around your pipeline calls
    from lexisearch.plugins.base import PluginContext
    ctx = PluginContext(operation="query", data={"query": "test"})
    registry.fire_before_query(ctx)
    # ... run query ...
    registry.fire_after_query(ctx)
"""

from __future__ import annotations

from lexisearch.plugins.base import (
    BasePlugin,
    PluginContext,
    PluginMeta,
    PluginState,
)
from lexisearch.plugins.registry import (
    PluginError,
    PluginRegistry,
    find_plugins_by_tag,
    get_registry,
    list_plugins,
    register_plugin,
    reset_registry,
)

__all__ = [
    # Base
    "BasePlugin",
    "PluginContext",
    # Registry
    "PluginError",
    "PluginMeta",
    "PluginRegistry",
    "PluginState",
    "find_plugins_by_tag",
    "get_registry",
    "list_plugins",
    "register_plugin",
    "reset_registry",
]
