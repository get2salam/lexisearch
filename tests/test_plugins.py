"""Tests for the plugin system."""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

from lexisearch.observability.metrics import InMemoryMetricsCollector, LexiMetrics
from lexisearch.plugins.base import BasePlugin, PluginContext, PluginMeta, PluginState
from lexisearch.plugins.builtin import (
    LoggingPlugin,
    MetricsPlugin,
    RateLimitExceeded,
    RateLimitPlugin,
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

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


class _EchoPlugin(BasePlugin):
    """Test plugin that records hook calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.loaded = False
        self.unloaded = False

    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="echo",
            version="0.1.0",
            description="Records hook calls for testing",
            tags=["test"],
        )

    def on_load(self, config: dict[str, Any] | None = None) -> None:
        self.loaded = True

    def on_unload(self) -> None:
        self.unloaded = True

    def on_before_query(self, ctx: PluginContext) -> None:
        self.calls.append("before_query")

    def on_after_query(self, ctx: PluginContext) -> None:
        self.calls.append("after_query")

    def on_before_ingest(self, ctx: PluginContext) -> None:
        self.calls.append("before_ingest")

    def on_after_ingest(self, ctx: PluginContext) -> None:
        self.calls.append("after_ingest")

    def on_before_retrieve(self, ctx: PluginContext) -> None:
        self.calls.append("before_retrieve")

    def on_after_retrieve(self, ctx: PluginContext) -> None:
        self.calls.append("after_retrieve")

    def on_before_generate(self, ctx: PluginContext) -> None:
        self.calls.append("before_generate")

    def on_after_generate(self, ctx: PluginContext) -> None:
        self.calls.append("after_generate")

    def on_error(self, ctx: PluginContext, exc: Exception) -> None:
        self.calls.append(f"error:{type(exc).__name__}")


class _FailPlugin(BasePlugin):
    """Plugin that raises on load for testing error state."""

    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(name="fail_plugin")

    def on_load(self, config: dict[str, Any] | None = None) -> None:
        raise RuntimeError("intentional load failure")


@pytest.fixture()
def registry() -> PluginRegistry:
    return PluginRegistry()


@pytest.fixture()
def echo(registry: PluginRegistry) -> _EchoPlugin:
    plugin = _EchoPlugin()
    registry.register(plugin, auto_load=True)
    return plugin


@pytest.fixture()
def ctx() -> PluginContext:
    return PluginContext(operation="query", data={"query": "test query"})


# ---------------------------------------------------------------------------
# Base plugin tests
# ---------------------------------------------------------------------------


class TestPluginMeta:
    def test_to_dict(self):
        meta = PluginMeta(
            name="my_plugin",
            version="1.2.3",
            description="Test",
            tags=["a", "b"],
            requires=["other"],
        )
        d = meta.to_dict()
        assert d["name"] == "my_plugin"
        assert d["version"] == "1.2.3"
        assert d["tags"] == ["a", "b"]


class TestPluginContext:
    def test_get_plugin_data_creates_on_first_access(self):
        ctx = PluginContext(operation="test")
        data = ctx.get_plugin_data("my_plugin")
        assert data == {}

    def test_get_plugin_data_is_persistent(self):
        ctx = PluginContext(operation="test")
        ctx.get_plugin_data("p")["key"] = "value"
        assert ctx.get_plugin_data("p")["key"] == "value"

    def test_get_plugin_data_isolated_per_plugin(self):
        ctx = PluginContext(operation="test")
        ctx.get_plugin_data("a")["x"] = 1
        ctx.get_plugin_data("b")["x"] = 2
        assert ctx.get_plugin_data("a")["x"] == 1
        assert ctx.get_plugin_data("b")["x"] == 2


class TestBasePlugin:
    def test_initial_state(self):
        plugin = _EchoPlugin()
        assert plugin.state == PluginState.REGISTERED

    def test_repr(self):
        plugin = _EchoPlugin()
        r = repr(plugin)
        assert "echo" in r
        assert "0.1.0" in r


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    def test_register_and_contains(self, registry: PluginRegistry):
        plugin = _EchoPlugin()
        registry.register(plugin)
        assert "echo" in registry

    def test_register_duplicate_raises(self, registry: PluginRegistry):
        registry.register(_EchoPlugin())
        with pytest.raises(PluginError, match="already registered"):
            registry.register(_EchoPlugin())

    def test_load_transitions_state(self, registry: PluginRegistry):
        plugin = _EchoPlugin()
        registry.register(plugin)
        assert plugin.state == PluginState.REGISTERED
        registry.load("echo")
        assert plugin.state == PluginState.ACTIVE
        assert plugin.loaded

    def test_auto_load(self, registry: PluginRegistry):
        plugin = _EchoPlugin()
        registry.register(plugin, auto_load=True)
        assert plugin.state == PluginState.ACTIVE

    def test_load_all(self, registry: PluginRegistry):
        p1 = _EchoPlugin()
        registry.register(p1)
        registry.load_all()
        assert p1.state == PluginState.ACTIVE

    def test_load_failure_sets_error_state(self, registry: PluginRegistry):
        plugin = _FailPlugin()
        registry.register(plugin)
        with pytest.raises(PluginError):
            registry.load("fail_plugin")
        assert plugin.state == PluginState.ERROR

    def test_unregister(self, registry: PluginRegistry):
        registry.register(_EchoPlugin(), auto_load=True)
        registry.unregister("echo")
        assert "echo" not in registry

    def test_unregister_calls_on_unload(self, registry: PluginRegistry):
        plugin = _EchoPlugin()
        registry.register(plugin, auto_load=True)
        registry.unregister("echo")
        assert plugin.unloaded

    def test_unregister_unknown_raises(self, registry: PluginRegistry):
        with pytest.raises(PluginError, match="not registered"):
            registry.unregister("nonexistent")

    def test_suspend_and_resume(
        self, registry: PluginRegistry, echo: _EchoPlugin, ctx: PluginContext
    ):
        registry.fire_before_query(ctx)
        assert "before_query" in echo.calls
        echo.calls.clear()

        registry.suspend("echo")
        registry.fire_before_query(ctx)
        assert "before_query" not in echo.calls

        registry.resume("echo")
        registry.fire_before_query(ctx)
        assert "before_query" in echo.calls

    def test_active_plugins(self, registry: PluginRegistry, echo: _EchoPlugin):
        assert echo in registry.active_plugins()

    def test_len(self, registry: PluginRegistry):
        assert len(registry) == 0
        registry.register(_EchoPlugin())
        assert len(registry) == 1

    def test_iter(self, registry: PluginRegistry):
        registry.register(_EchoPlugin())
        plugins = list(registry)
        assert len(plugins) == 1

    def test_get(self, registry: PluginRegistry, echo: _EchoPlugin):
        assert registry.get("echo") is echo
        assert registry.get("missing") is None

    def test_describe(self, registry: PluginRegistry, echo: _EchoPlugin):
        desc = registry.describe()
        assert len(desc) == 1
        assert desc[0]["name"] == "echo"
        assert "state" in desc[0]

    def test_hook_dispatch_before_after_query(
        self, registry: PluginRegistry, echo: _EchoPlugin, ctx: PluginContext
    ):
        registry.fire_before_query(ctx)
        registry.fire_after_query(ctx)
        assert echo.calls == ["before_query", "after_query"]

    def test_hook_dispatch_ingest(self, registry: PluginRegistry, echo: _EchoPlugin):
        ctx = PluginContext(operation="ingest", data={"documents": []})
        registry.fire_before_ingest(ctx)
        registry.fire_after_ingest(ctx)
        assert "before_ingest" in echo.calls
        assert "after_ingest" in echo.calls

    def test_hook_dispatch_retrieve(
        self, registry: PluginRegistry, echo: _EchoPlugin, ctx: PluginContext
    ):
        registry.fire_before_retrieve(ctx)
        registry.fire_after_retrieve(ctx)
        assert "before_retrieve" in echo.calls
        assert "after_retrieve" in echo.calls

    def test_hook_dispatch_generate(
        self, registry: PluginRegistry, echo: _EchoPlugin, ctx: PluginContext
    ):
        registry.fire_before_generate(ctx)
        registry.fire_after_generate(ctx)
        assert "before_generate" in echo.calls
        assert "after_generate" in echo.calls

    def test_hook_dispatch_error(
        self, registry: PluginRegistry, echo: _EchoPlugin, ctx: PluginContext
    ):
        exc = ValueError("test error")
        registry.fire_error(ctx, exc)
        assert "error:ValueError" in echo.calls

    def test_hook_exception_does_not_propagate(self, registry: PluginRegistry, ctx: PluginContext):
        """A buggy plugin should not crash the caller."""

        class BuggyPlugin(BasePlugin):
            @property
            def meta(self) -> PluginMeta:
                return PluginMeta(name="buggy")

            def on_before_query(self, ctx: PluginContext) -> None:
                raise RuntimeError("plugin bug")

        plugin = BuggyPlugin()
        registry.register(plugin, auto_load=True)
        # Should not raise
        registry.fire_before_query(ctx)


# ---------------------------------------------------------------------------
# Global registry helpers
# ---------------------------------------------------------------------------


class TestGlobalRegistryHelpers:
    def setup_method(self):
        reset_registry()

    def teardown_method(self):
        reset_registry()

    def test_register_plugin_global(self):
        register_plugin(_EchoPlugin())
        plugins = list_plugins()
        assert any(p["name"] == "echo" for p in plugins)

    def test_get_registry_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_find_plugins_by_tag(self):
        reset_registry()
        register_plugin(_EchoPlugin())
        results = find_plugins_by_tag("test")
        assert any(m.name == "echo" for m in results)

    def test_find_plugins_by_tag_no_match(self):
        reset_registry()
        register_plugin(_EchoPlugin())
        results = find_plugins_by_tag("nonexistent_tag")
        assert results == []


# ---------------------------------------------------------------------------
# Built-in plugin tests
# ---------------------------------------------------------------------------


class TestLoggingPlugin:
    def test_meta(self):
        p = LoggingPlugin()
        assert p.meta.name == "logging"
        assert "builtin" in p.meta.tags

    def test_lifecycle(self):
        registry = PluginRegistry()
        plugin = LoggingPlugin()
        registry.register(plugin, auto_load=True)
        assert plugin.state == PluginState.ACTIVE

    def test_query_hooks_do_not_raise(self, caplog):
        with caplog.at_level(logging.INFO, logger="lexisearch.pipeline"):
            registry = PluginRegistry()
            registry.register(LoggingPlugin(), auto_load=True)
            ctx = PluginContext(operation="query", data={"query": "hello world"})
            registry.fire_before_query(ctx)
            ctx.data["results"] = [1, 2, 3]
            registry.fire_after_query(ctx)

    def test_ingest_hooks_do_not_raise(self):
        registry = PluginRegistry()
        registry.register(LoggingPlugin(), auto_load=True)
        ctx = PluginContext(
            operation="ingest",
            data={"documents": ["doc1", "doc2"], "chunks": list(range(10))},
        )
        registry.fire_before_ingest(ctx)
        registry.fire_after_ingest(ctx)

    def test_error_hook_does_not_raise(self):
        registry = PluginRegistry()
        registry.register(LoggingPlugin(), auto_load=True)
        ctx = PluginContext(operation="query")
        registry.fire_error(ctx, RuntimeError("test"))


class TestMetricsPlugin:
    def test_meta(self):
        p = MetricsPlugin()
        assert p.meta.name == "metrics"

    def test_query_increments_counter(self):
        collector = InMemoryMetricsCollector()
        registry = PluginRegistry()
        registry.register(MetricsPlugin(collector=collector), auto_load=True)

        ctx = PluginContext(operation="query", data={"query": "test"})
        registry.fire_before_query(ctx)
        time.sleep(0.001)
        registry.fire_after_query(ctx)

        assert collector.get_counter(LexiMetrics.QUERIES_TOTAL) == 1.0
        summary = collector.summarize_histogram(LexiMetrics.QUERY_LATENCY_MS)
        assert summary is not None
        assert summary.count == 1

    def test_ingest_records_counts(self):
        collector = InMemoryMetricsCollector()
        registry = PluginRegistry()
        registry.register(MetricsPlugin(collector=collector), auto_load=True)

        ctx = PluginContext(
            operation="ingest",
            data={"documents": ["a", "b", "c"], "chunks": list(range(9))},
        )
        registry.fire_before_ingest(ctx)
        time.sleep(0.001)
        registry.fire_after_ingest(ctx)

        assert collector.get_counter(LexiMetrics.DOCUMENTS_INGESTED) == 3.0
        assert collector.get_counter(LexiMetrics.CHUNKS_CREATED) == 9.0

    def test_error_increments_error_counter(self):
        collector = InMemoryMetricsCollector()
        registry = PluginRegistry()
        registry.register(MetricsPlugin(collector=collector), auto_load=True)

        ctx = PluginContext(operation="query")
        registry.fire_error(ctx, ValueError("oops"))

        assert (
            collector.get_counter(
                LexiMetrics.ERRORS_TOTAL, operation="query", error_type="ValueError"
            )
            == 1.0
        )


class TestRateLimitPlugin:
    def test_meta(self):
        p = RateLimitPlugin()
        assert p.meta.name == "rate_limit"

    def test_tokens_available_after_init(self):
        plugin = RateLimitPlugin(rate=10.0, burst=10.0)
        assert plugin.available_tokens == 10.0

    def test_consumes_token_per_query(self):
        plugin = RateLimitPlugin(rate=100.0, burst=10.0)
        initial = plugin.available_tokens
        registry = PluginRegistry()
        registry.register(plugin, auto_load=True)
        ctx = PluginContext(operation="query")
        registry.fire_before_query(ctx)
        assert plugin.available_tokens < initial

    def test_raises_when_rate_exceeded(self):
        plugin = RateLimitPlugin(rate=1.0, burst=1.0, raise_on_exceed=True)
        registry = PluginRegistry()
        registry.register(plugin, auto_load=True)
        ctx = PluginContext(operation="query")
        # First call should succeed
        registry.fire_before_query(ctx)
        # Second call should exceed rate
        # Note: dispatch catches exceptions — we test the plugin directly
        with pytest.raises(RateLimitExceeded):
            plugin.on_before_query(ctx)

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError):
            RateLimitPlugin(rate=0.0)

    def test_on_load_config_override(self):
        plugin = RateLimitPlugin(rate=5.0)
        plugin.on_load({"rate": 20.0, "burst": 50.0})
        assert plugin._rate == 20.0
        assert plugin._burst == 50.0

    def test_rate_refill_over_time(self):
        plugin = RateLimitPlugin(rate=1000.0, burst=1.0, raise_on_exceed=True)
        ctx = PluginContext(operation="query")
        # Consume the one token
        plugin.on_before_query(ctx)
        # After a small sleep, should have refilled
        time.sleep(0.002)
        # Should not raise now
        plugin.on_before_query(ctx)
