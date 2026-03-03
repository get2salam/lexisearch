"""Plugin registry — load, discover, and manage plugins."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from lexisearch.plugins.base import BasePlugin, PluginMeta, PluginState

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

P = TypeVar("P", bound=BasePlugin)


class PluginError(Exception):
    """Raised when a plugin operation fails."""


class PluginRegistry:
    """Central registry for LexiSearch plugins.

    Manages plugin lifecycle (load → active → unload) and provides
    discovery helpers for scanning modules and directories.

    Example::

        registry = PluginRegistry()
        registry.register(TimingPlugin())
        registry.load_all()

        # Fire hooks on all active plugins
        registry.fire_before_query(ctx)
        # ... run query ...
        registry.fire_after_query(ctx)
    """

    def __init__(self) -> None:
        """Initialise an empty plugin registry."""
        self._plugins: dict[str, BasePlugin] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        plugin: BasePlugin,
        *,
        config: dict[str, Any] | None = None,
        auto_load: bool = False,
    ) -> None:
        """Register a plugin instance.

        Args:
            plugin: The plugin to register.
            config: Optional configuration dict passed to :meth:`on_load`.
            auto_load: If ``True``, immediately load the plugin.

        Raises:
            PluginError: If a plugin with the same name is already registered.
        """
        name = plugin.meta.name
        if name in self._plugins:
            raise PluginError(f"Plugin '{name}' is already registered")
        self._plugins[name] = plugin
        logger.debug("Registered plugin '%s' v%s", name, plugin.meta.version)
        if auto_load:
            self.load(name, config=config)

    def unregister(self, name: str) -> None:
        """Unregister and unload a plugin.

        Args:
            name: Plugin name to remove.

        Raises:
            PluginError: If no plugin with *name* is registered.
        """
        if name not in self._plugins:
            raise PluginError(f"Plugin '{name}' is not registered")
        plugin = self._plugins.pop(name)
        try:
            plugin.on_unload()
            plugin._state = PluginState.UNLOADED
        except Exception as exc:
            logger.warning("Error unloading plugin '%s': %s", name, exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, name: str, *, config: dict[str, Any] | None = None) -> None:
        """Load a registered plugin.

        Args:
            name: Plugin name to load.
            config: Optional configuration forwarded to :meth:`on_load`.

        Raises:
            PluginError: If the plugin is not registered.
        """
        plugin = self._get(name)
        try:
            plugin.on_load(config)
            plugin._state = PluginState.ACTIVE
            logger.info("Loaded plugin '%s'", name)
        except Exception as exc:
            plugin._state = PluginState.ERROR
            raise PluginError(f"Failed to load plugin '{name}': {exc}") from exc

    def load_all(self, *, config: dict[str, dict[str, Any]] | None = None) -> None:
        """Load all registered plugins that are not yet active.

        Args:
            config: Optional ``{plugin_name: config_dict}`` map.
        """
        for name, plugin in self._plugins.items():
            if plugin.state == PluginState.REGISTERED:
                plugin_config = (config or {}).get(name)
                self.load(name, config=plugin_config)

    def suspend(self, name: str) -> None:
        """Temporarily suspend a plugin (hooks will be skipped).

        Args:
            name: Plugin name to suspend.
        """
        plugin = self._get(name)
        plugin._state = PluginState.SUSPENDED

    def resume(self, name: str) -> None:
        """Resume a suspended plugin.

        Args:
            name: Plugin name to resume.
        """
        plugin = self._get(name)
        if plugin.state == PluginState.SUSPENDED:
            plugin._state = PluginState.ACTIVE

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get(self, name: str) -> BasePlugin | None:
        """Return a plugin by name, or ``None`` if not found.

        Args:
            name: Plugin name.

        Returns:
            The plugin instance, or ``None``.
        """
        return self._plugins.get(name)

    def get_typed(self, name: str, plugin_type: type[P]) -> P | None:
        """Return a plugin cast to *plugin_type*, or ``None``.

        Args:
            name: Plugin name.
            plugin_type: Expected plugin class.

        Returns:
            The typed plugin, or ``None``.
        """
        plugin = self._plugins.get(name)
        if isinstance(plugin, plugin_type):
            return plugin
        return None

    def active_plugins(self) -> list[BasePlugin]:
        """Return all currently active plugins.

        Returns:
            List of active :class:`BasePlugin` instances.
        """
        return [p for p in self._plugins.values() if p.state == PluginState.ACTIVE]

    def all_plugins(self) -> list[BasePlugin]:
        """Return all registered plugins regardless of state.

        Returns:
            List of all :class:`BasePlugin` instances.
        """
        return list(self._plugins.values())

    def __iter__(self) -> Iterator[BasePlugin]:
        """Iterate over registered plugins."""
        return iter(self._plugins.values())

    def __len__(self) -> int:
        """Return the number of registered plugins."""
        return len(self._plugins)

    def __contains__(self, name: object) -> bool:
        """Check if a plugin name is registered."""
        return name in self._plugins

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def load_from_module(self, module_path: str) -> list[str]:
        """Discover and register all :class:`BasePlugin` subclasses in a module.

        Args:
            module_path: Dotted Python module path (e.g.
                ``"myapp.plugins.auth"``).

        Returns:
            Names of newly registered plugins.

        Raises:
            PluginError: If the module cannot be imported.
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise PluginError(f"Cannot import module '{module_path}': {exc}") from exc

        registered: list[str] = []
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BasePlugin)
                and obj is not BasePlugin
                and not inspect.isabstract(obj)
            ):
                try:
                    instance = obj()
                    self.register(instance)
                    registered.append(instance.meta.name)
                except PluginError:
                    pass  # Already registered
        return registered

    def load_from_directory(self, directory: str | Path) -> list[str]:
        """Discover plugins in all ``*.py`` files under *directory*.

        Args:
            directory: Filesystem path to scan.

        Returns:
            Names of newly registered plugins.
        """
        path = Path(directory)
        if not path.is_dir():
            raise PluginError(f"'{directory}' is not a directory")

        registered: list[str] = []
        for py_file in path.rglob("*.py"):
            module_name = py_file.stem
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                logger.warning("Skipping '%s': %s", py_file, exc)
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BasePlugin)
                    and obj is not BasePlugin
                    and not inspect.isabstract(obj)
                ):
                    try:
                        instance = obj()
                        self.register(instance)
                        registered.append(instance.meta.name)
                    except PluginError:
                        pass
        return registered

    # ------------------------------------------------------------------
    # Hook dispatch
    # ------------------------------------------------------------------

    def _get(self, name: str) -> BasePlugin:
        plugin = self._plugins.get(name)
        if plugin is None:
            raise PluginError(f"Plugin '{name}' is not registered")
        return plugin

    def _dispatch(self, hook: str, *args: Any, **kwargs: Any) -> None:
        """Call *hook* on every active plugin, logging individual failures.

        Args:
            hook: Method name to call on each plugin.
            *args: Positional args forwarded to the hook.
            **kwargs: Keyword args forwarded to the hook.
        """
        for plugin in self.active_plugins():
            method = getattr(plugin, hook, None)
            if method is not None:
                try:
                    method(*args, **kwargs)
                except Exception as exc:
                    logger.error(
                        "Plugin '%s' raised in '%s': %s",
                        plugin.meta.name,
                        hook,
                        exc,
                    )

    # Typed dispatch helpers -----------------------------------------------

    def fire_before_query(self, ctx: Any) -> None:
        """Fire the ``on_before_query`` hook on all active plugins."""
        self._dispatch("on_before_query", ctx)

    def fire_after_query(self, ctx: Any) -> None:
        """Fire the ``on_after_query`` hook on all active plugins."""
        self._dispatch("on_after_query", ctx)

    def fire_before_ingest(self, ctx: Any) -> None:
        """Fire the ``on_before_ingest`` hook on all active plugins."""
        self._dispatch("on_before_ingest", ctx)

    def fire_after_ingest(self, ctx: Any) -> None:
        """Fire the ``on_after_ingest`` hook on all active plugins."""
        self._dispatch("on_after_ingest", ctx)

    def fire_before_retrieve(self, ctx: Any) -> None:
        """Fire the ``on_before_retrieve`` hook on all active plugins."""
        self._dispatch("on_before_retrieve", ctx)

    def fire_after_retrieve(self, ctx: Any) -> None:
        """Fire the ``on_after_retrieve`` hook on all active plugins."""
        self._dispatch("on_after_retrieve", ctx)

    def fire_before_generate(self, ctx: Any) -> None:
        """Fire the ``on_before_generate`` hook on all active plugins."""
        self._dispatch("on_before_generate", ctx)

    def fire_after_generate(self, ctx: Any) -> None:
        """Fire the ``on_after_generate`` hook on all active plugins."""
        self._dispatch("on_after_generate", ctx)

    def fire_error(self, ctx: Any, exc: Exception) -> None:
        """Fire the ``on_error`` hook on all active plugins."""
        self._dispatch("on_error", ctx, exc)

    def describe(self) -> list[dict[str, Any]]:
        """Return metadata for all registered plugins.

        Returns:
            List of :meth:`PluginMeta.to_dict` dicts, one per plugin.
        """
        return [{**p.meta.to_dict(), "state": p.state.value} for p in self._plugins.values()]


# ---------------------------------------------------------------------------
# Module-level global registry
# ---------------------------------------------------------------------------


_global_registry: PluginRegistry | None = None


def get_registry() -> PluginRegistry:
    """Return the process-wide plugin registry (created lazily).

    Returns:
        The singleton :class:`PluginRegistry` instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the process-wide registry (useful in tests).

    Returns:
        None.
    """
    global _global_registry
    _global_registry = None


def list_plugins() -> list[dict[str, Any]]:
    """Return metadata for all plugins in the global registry.

    Returns:
        List of plugin metadata dicts.
    """
    return get_registry().describe()


def register_plugin(
    plugin: BasePlugin,
    *,
    config: dict[str, Any] | None = None,
    auto_load: bool = True,
) -> None:
    """Register and optionally load a plugin in the global registry.

    Args:
        plugin: The plugin instance to register.
        config: Optional configuration forwarded to :meth:`on_load`.
        auto_load: If ``True`` (default), immediately load the plugin.
    """
    get_registry().register(plugin, config=config, auto_load=auto_load)


def find_plugins_by_tag(tag: str) -> list[PluginMeta]:
    """Return metadata for all registered plugins that have *tag*.

    Args:
        tag: Tag string to filter by.

    Returns:
        List of :class:`PluginMeta` for matching plugins.
    """
    return [p.meta for p in get_registry() if tag in p.meta.tags]
