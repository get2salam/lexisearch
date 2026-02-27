"""Tests for the LexiSearch pipeline orchestration layer.

Covers: events, config, registry, builder, runner — all five modules.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from lexisearch.embeddings import MockEmbedder
from lexisearch.generation import MockLLM
from lexisearch.models import Document
from lexisearch.pipeline.builder import BuiltPipeline, PipelineBuilder, PipelineBuilderError
from lexisearch.pipeline.config import (
    ChunkConfig,
    ChunkMethod,
    PipelineConfig,
    RetrieveConfig,
    RetrieveMethod,
    default_config,
    from_dict,
    load_config,
)
from lexisearch.pipeline.events import (
    EventBus,
    EventError,
    EventType,
    PipelineEvent,
)
from lexisearch.pipeline.registry import (
    ComponentInfo,
    ComponentKind,
    ComponentRegistry,
    RegistryError,
    discover_plugins,
)
from lexisearch.pipeline.runner import (
    IngestResult,
    PipelineError,
    PipelineRunner,
    QueryResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder() -> MockEmbedder:
    return MockEmbedder(dimensions=32)


@pytest.fixture
def llm() -> MockLLM:
    return MockLLM(response_text="The answer is 42.")


@pytest.fixture
def built_pipeline(embedder: MockEmbedder, llm: MockLLM) -> BuiltPipeline:
    return PipelineBuilder.create("test").embed(embedder).store().retrieve().generate(llm).build()


@pytest.fixture
def runner(built_pipeline: BuiltPipeline) -> PipelineRunner:
    return PipelineRunner(built_pipeline)


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(content="Dense retrieval uses neural embeddings for semantic search."),
        Document(content="BM25 is a classic sparse retrieval algorithm based on TF-IDF."),
        Document(content="Hybrid search combines dense and sparse retrieval methods."),
    ]


# ---------------------------------------------------------------------------
# Events — EventBus
# ---------------------------------------------------------------------------


class TestEventBus:
    def test_subscribe_and_emit(self):
        bus = EventBus()
        received: list[PipelineEvent] = []
        bus.subscribe(received.append)
        event = PipelineEvent(EventType.PIPELINE_START, "p1")
        bus.emit(event)
        assert len(received) == 1
        assert received[0].event_type is EventType.PIPELINE_START

    def test_filter_by_event_type(self):
        bus = EventBus()
        starts: list[PipelineEvent] = []
        bus.subscribe(starts.append, EventType.PIPELINE_START)
        bus.emit(PipelineEvent(EventType.PIPELINE_START, "p1"))
        bus.emit(PipelineEvent(EventType.PIPELINE_FINISH, "p1"))
        assert len(starts) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received: list[PipelineEvent] = []

        def handler(event: PipelineEvent) -> None:
            received.append(event)

        bus.subscribe(handler)
        removed = bus.unsubscribe(handler)
        bus.emit(PipelineEvent(EventType.PIPELINE_START, "p1"))
        assert removed == 1
        assert len(received) == 0

    def test_clear(self):
        bus = EventBus()
        bus.subscribe(lambda e: None)
        bus.subscribe(lambda e: None)
        bus.clear()
        assert bus.handler_count == 0

    def test_handler_count(self):
        bus = EventBus()
        bus.subscribe(lambda e: None)
        bus.subscribe(lambda e: None)
        assert bus.handler_count == 2

    def test_raise_on_error(self):
        bus = EventBus(raise_on_error=True)

        def bad_handler(event: PipelineEvent) -> None:
            raise ValueError("boom")

        bus.subscribe(bad_handler)
        with pytest.raises(EventError) as exc_info:
            bus.emit(PipelineEvent(EventType.PIPELINE_START, "p1"))
        assert "boom" in str(exc_info.value)

    def test_no_raise_on_error(self):
        bus = EventBus(raise_on_error=False)

        def bad_handler(event: PipelineEvent) -> None:
            raise ValueError("boom")

        received: list[PipelineEvent] = []
        bus.subscribe(bad_handler)
        bus.subscribe(received.append)  # should still fire
        bus.emit(PipelineEvent(EventType.PIPELINE_START, "p1"))
        assert len(received) == 1

    def test_convenience_emit_start(self):
        bus = EventBus()
        events: list[PipelineEvent] = []
        bus.subscribe(events.append)
        bus.emit_start("p1", {"run_id": "abc"})
        assert events[0].event_type is EventType.PIPELINE_START
        assert events[0].data["run_id"] == "abc"

    def test_convenience_emit_finish(self):
        bus = EventBus()
        events: list[PipelineEvent] = []
        bus.subscribe(events.append)
        bus.emit_finish("p1")
        assert events[0].event_type is EventType.PIPELINE_FINISH

    def test_convenience_emit_error(self):
        bus = EventBus()
        events: list[PipelineEvent] = []
        bus.subscribe(events.append)
        bus.emit_error("p1", ValueError("err"), step="chunk")
        assert events[0].event_type is EventType.PIPELINE_ERROR
        assert events[0].step == "chunk"
        assert isinstance(events[0].error, ValueError)

    def test_convenience_emit_progress(self):
        bus = EventBus()
        events: list[PipelineEvent] = []
        bus.subscribe(events.append)
        bus.emit_progress("p1", "embed", 5, 10)
        assert events[0].event_type is EventType.PROGRESS
        assert events[0].data["current"] == 5
        assert events[0].data["total"] == 10

    def test_pipeline_event_repr(self):
        event = PipelineEvent(EventType.STEP_START, "pipe1", step="chunk")
        assert "step_start" in repr(event)
        assert "chunk" in repr(event)

    def test_event_bus_repr(self):
        bus = EventBus()
        bus.subscribe(lambda e: None)
        assert "handlers=1" in repr(bus)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestChunkConfig:
    def test_default_values(self):
        cfg = ChunkConfig()
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64
        assert cfg.method is ChunkMethod.RECURSIVE

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkConfig(chunk_size=0)

    def test_invalid_overlap_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            ChunkConfig(chunk_overlap=-1)

    def test_overlap_gte_size(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkConfig(chunk_size=100, chunk_overlap=100)


class TestRetrieveConfig:
    def test_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            RetrieveConfig(top_k=0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            RetrieveConfig(alpha=1.5)


class TestPipelineConfig:
    def test_to_dict(self):
        cfg = PipelineConfig(name="myp")
        d = cfg.to_dict()
        assert d["name"] == "myp"
        assert "chunk" in d
        assert d["chunk"]["method"] == "recursive"

    def test_to_json(self):
        cfg = PipelineConfig(name="json-test")
        j = cfg.to_json()
        parsed = json.loads(j)
        assert parsed["name"] == "json-test"

    def test_save_and_load(self, tmp_path: Path):
        cfg = PipelineConfig(name="saved")
        fpath = tmp_path / "pipeline.json"
        cfg.save(fpath)
        loaded = load_config(fpath)
        assert loaded.name == "saved"

    def test_load_config_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "missing.json")

    def test_from_dict(self):
        d = {
            "name": "from_dict_test",
            "chunk": {"method": "sentence", "chunk_size": 256, "chunk_overlap": 32},
        }
        cfg = from_dict(d)
        assert cfg.name == "from_dict_test"
        assert cfg.chunk.method is ChunkMethod.SENTENCE
        assert cfg.chunk.chunk_size == 256

    def test_from_dict_unknown_enum_value(self):
        d = {"chunk": {"method": "unknown_method"}}
        cfg = from_dict(d)  # should not raise; unknown kept as raw
        assert cfg.chunk.method == "unknown_method"


class TestDefaultConfig:
    def test_rag_preset(self):
        cfg = default_config("rag")
        assert cfg.name == "rag"
        assert cfg.retrieve.method is RetrieveMethod.HYBRID

    def test_qa_preset(self):
        cfg = default_config("qa")
        assert cfg.retrieve.method is RetrieveMethod.VECTOR

    def test_summarise_preset(self):
        cfg = default_config("summarise")
        assert cfg.retrieve.method is RetrieveMethod.BM25

    def test_unknown_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            default_config("nonexistent")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestComponentRegistry:
    def test_register_and_get(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        reg.register(ComponentKind.LLM, "echo", MockLLM, "Echo LLM")
        info = reg.get(ComponentKind.LLM, "echo")
        assert info.alias == "echo"
        assert info.factory is MockLLM

    def test_register_duplicate_raises(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        reg.register(ComponentKind.LLM, "x", MockLLM)
        with pytest.raises(RegistryError):
            reg.register(ComponentKind.LLM, "x", MockLLM)

    def test_register_overwrite(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        reg.register(ComponentKind.LLM, "x", MockLLM)
        reg.register(ComponentKind.LLM, "x", MockEmbedder, overwrite=True)
        info = reg.get(ComponentKind.LLM, "x")
        assert info.factory is MockEmbedder

    def test_unregister(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        reg.register(ComponentKind.LLM, "y", MockLLM)
        removed = reg.unregister(ComponentKind.LLM, "y")
        assert removed is True
        assert not reg.has(ComponentKind.LLM, "y")

    def test_unregister_missing(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        assert reg.unregister(ComponentKind.LLM, "nope") is False

    def test_get_missing_raises(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        with pytest.raises(RegistryError, match=r"No llm registered"):
            reg.get(ComponentKind.LLM, "missing")

    def test_has(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        reg.register(ComponentKind.EMBEDDER, "mock", MockEmbedder)
        assert reg.has(ComponentKind.EMBEDDER, "mock") is True
        assert reg.has(ComponentKind.EMBEDDER, "nope") is False

    def test_list_aliases(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        reg.register(ComponentKind.CHUNKER, "b", MockLLM)
        reg.register(ComponentKind.CHUNKER, "a", MockLLM)
        aliases = reg.list_aliases(ComponentKind.CHUNKER)
        assert aliases == ["a", "b"]

    def test_list_all(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        result = reg.list_all()
        assert set(result.keys()) == {k.value for k in ComponentKind}

    def test_build_calls_factory(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        reg.register(ComponentKind.EMBEDDER, "mock", lambda **kw: MockEmbedder(**kw))
        emb = reg.build(ComponentKind.EMBEDDER, "mock", dimensions=64)
        assert emb.dimensions() == 64

    def test_builtin_loaders_registered(self):
        reg = ComponentRegistry(auto_register_builtins=True)
        assert reg.has(ComponentKind.LOADER, "text")
        assert reg.has(ComponentKind.LOADER, "pdf")
        assert reg.has(ComponentKind.LOADER, "html")

    def test_builtin_llms_registered(self):
        reg = ComponentRegistry(auto_register_builtins=True)
        assert reg.has(ComponentKind.LLM, "mock")
        assert reg.has(ComponentKind.LLM, "openai")

    def test_register_many(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        infos = [
            ComponentInfo(ComponentKind.LLM, "a", MockLLM, "A"),
            ComponentInfo(ComponentKind.LLM, "b", MockLLM, "B"),
        ]
        reg.register_many(infos)
        assert reg.has(ComponentKind.LLM, "a")
        assert reg.has(ComponentKind.LLM, "b")

    def test_registry_repr(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        assert "ComponentRegistry" in repr(reg)

    def test_discover_plugins_no_dir(self):
        reg = ComponentRegistry(auto_register_builtins=False)
        added = discover_plugins(reg, plugins_dir=None)
        assert added == 0

    def test_discover_plugins_empty_dir(self, tmp_path: Path):
        reg = ComponentRegistry(auto_register_builtins=False)
        added = discover_plugins(reg, plugins_dir=str(tmp_path))
        assert added == 0

    def test_discover_plugins_from_file(self, tmp_path: Path):
        plugin_code = """
from lexisearch.pipeline.registry import ComponentRegistry, ComponentKind, ComponentInfo
from lexisearch.generation import MockLLM

registry = ComponentRegistry(auto_register_builtins=False)
registry.register(ComponentKind.LLM, "plugin-llm", MockLLM, "Plugin LLM")
"""
        (tmp_path / "my_plugin.py").write_text(plugin_code, encoding="utf-8")
        reg = ComponentRegistry(auto_register_builtins=False)
        added = discover_plugins(reg, plugins_dir=str(tmp_path))
        assert added == 1
        assert reg.has(ComponentKind.LLM, "plugin-llm")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TestPipelineBuilder:
    def test_create_class_method(self):
        builder = PipelineBuilder.create("myp")
        assert isinstance(builder, PipelineBuilder)

    def test_build_missing_embedder(self, llm: MockLLM):
        with pytest.raises(PipelineBuilderError, match="embed"):
            PipelineBuilder.create("bad").generate(llm).build()

    def test_build_missing_llm(self, embedder: MockEmbedder):
        with pytest.raises(PipelineBuilderError, match="generate"):
            PipelineBuilder.create("bad").embed(embedder).build()

    def test_build_returns_built_pipeline(self, embedder: MockEmbedder, llm: MockLLM):
        pipeline = PipelineBuilder.create("ok").embed(embedder).generate(llm).build()
        assert isinstance(pipeline, BuiltPipeline)

    def test_pipeline_id_prefix(self, embedder: MockEmbedder, llm: MockLLM):
        pipeline = PipelineBuilder.create("myname").embed(embedder).generate(llm).build()
        assert pipeline.pipeline_id.startswith("myname-")

    def test_explicit_pipeline_id(self, embedder: MockEmbedder, llm: MockLLM):
        pipeline = (
            PipelineBuilder("p", pipeline_id="fixed-id").embed(embedder).generate(llm).build()
        )
        assert pipeline.pipeline_id == "fixed-id"

    def test_with_events(self, embedder: MockEmbedder, llm: MockLLM):
        bus = EventBus()
        pipeline = (
            PipelineBuilder.create("ev").embed(embedder).generate(llm).with_events(bus).build()
        )
        assert pipeline.event_bus is bus

    def test_with_metadata(self, embedder: MockEmbedder, llm: MockLLM):
        pipeline = (
            PipelineBuilder.create("meta")
            .embed(embedder)
            .generate(llm)
            .with_metadata(env="test", version=2)
            .build()
        )
        assert pipeline.metadata["env"] == "test"
        assert pipeline.metadata["version"] == 2

    def test_chunk_config_propagated(self, embedder: MockEmbedder, llm: MockLLM):
        pipeline = (
            PipelineBuilder.create("chk")
            .chunk(chunk_size=128, chunk_overlap=16)
            .embed(embedder)
            .generate(llm)
            .build()
        )
        assert pipeline.config.chunk.chunk_size == 128

    def test_retrieve_config_propagated(self, embedder: MockEmbedder, llm: MockLLM):
        pipeline = (
            PipelineBuilder.create("ret").embed(embedder).retrieve(top_k=7).generate(llm).build()
        )
        assert pipeline.config.retrieve.top_k == 7

    def test_validate_returns_errors(self):
        builder = PipelineBuilder.create("bad")
        errors = builder.validate()
        assert len(errors) >= 2  # embedder + llm

    def test_repr(self):
        builder = PipelineBuilder.create("test")
        assert "PipelineBuilder" in repr(builder)

    def test_built_pipeline_repr(self, built_pipeline: BuiltPipeline):
        r = repr(built_pipeline)
        assert "BuiltPipeline" in r
        assert built_pipeline.pipeline_id in r


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestPipelineRunner:
    def test_ingest_documents(self, runner: PipelineRunner, sample_docs: list[Document]):
        result = runner.ingest_documents(sample_docs)
        assert isinstance(result, IngestResult)
        assert result.document_count == 3
        assert result.chunk_count >= 3
        assert result.embedded_count == result.chunk_count
        assert result.success is True

    def test_ingest_result_latency(self, runner: PipelineRunner, sample_docs: list[Document]):
        result = runner.ingest_documents(sample_docs)
        assert result.latency_ms > 0

    def test_ingest_stage_latencies(self, runner: PipelineRunner, sample_docs: list[Document]):
        result = runner.ingest_documents(sample_docs)
        assert "chunk" in result.stage_latencies
        assert "embed" in result.stage_latencies
        assert "index" in result.stage_latencies

    def test_query_returns_result(self, runner: PipelineRunner, sample_docs: list[Document]):
        runner.ingest_documents(sample_docs)
        result = runner.query("What is dense retrieval?")
        assert isinstance(result, QueryResult)
        assert result.query == "What is dense retrieval?"
        assert len(result.answer) > 0

    def test_query_sources(self, runner: PipelineRunner, sample_docs: list[Document]):
        runner.ingest_documents(sample_docs)
        result = runner.query("dense retrieval")
        assert isinstance(result.sources, list)

    def test_query_latencies(self, runner: PipelineRunner, sample_docs: list[Document]):
        runner.ingest_documents(sample_docs)
        result = runner.query("BM25")
        assert result.retrieval_latency_ms >= 0
        assert result.generation_latency_ms >= 0
        assert result.total_latency_ms >= 0

    def test_progress_callback(self, built_pipeline: BuiltPipeline, sample_docs: list[Document]):
        calls: list[tuple[str, int, int]] = []

        def cb(step: str, current: int, total: int, data: dict[str, Any]) -> None:
            calls.append((step, current, total))

        rr = PipelineRunner(built_pipeline, progress_callback=cb)
        rr.ingest_documents(sample_docs)
        assert len(calls) > 0
        steps = {c[0] for c in calls}
        assert "chunk" in steps or "embed" in steps

    def test_event_bus_fires_events(
        self, embedder: MockEmbedder, llm: MockLLM, sample_docs: list[Document]
    ):
        bus = EventBus()
        fired: list[EventType] = []
        bus.subscribe(lambda e: fired.append(e.event_type))

        pipeline = (
            PipelineBuilder.create("ev")
            .embed(embedder)
            .store()
            .retrieve()
            .generate(llm)
            .with_events(bus)
            .build()
        )
        rr = PipelineRunner(pipeline)
        rr.ingest_documents(sample_docs)
        assert EventType.PIPELINE_START in fired
        assert EventType.PIPELINE_FINISH in fired

    def test_ingest_empty_documents(self, runner: PipelineRunner):
        result = runner.ingest_documents([])
        assert result.document_count == 0
        assert result.chunk_count == 0

    def test_ingest_from_loader_no_loader(self, runner: PipelineRunner):
        with pytest.raises(PipelineError, match="No loader"):
            runner.ingest_from_loader("some/path.txt")

    def test_pipeline_runner_repr(self, runner: PipelineRunner):
        assert "PipelineRunner" in repr(runner)

    def test_query_result_repr(self, runner: PipelineRunner, sample_docs: list[Document]):
        runner.ingest_documents(sample_docs)
        result = runner.query("hybrid search")
        assert "QueryResult" in repr(result)

    def test_ingest_result_repr(self, runner: PipelineRunner, sample_docs: list[Document]):
        result = runner.ingest_documents(sample_docs)
        assert "IngestResult" in repr(result)


# ---------------------------------------------------------------------------
# Async Runner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_ingest_documents(built_pipeline: BuiltPipeline, sample_docs: list[Document]):
    rr = PipelineRunner(built_pipeline)
    result = await rr.aingest_documents(sample_docs)
    assert result.document_count == 3
    assert result.embedded_count >= 3


@pytest.mark.asyncio
async def test_async_query(built_pipeline: BuiltPipeline, sample_docs: list[Document]):
    rr = PipelineRunner(built_pipeline)
    await rr.aingest_documents(sample_docs)
    result = await rr.aquery("What is hybrid search?")
    assert isinstance(result, QueryResult)
    assert len(result.answer) > 0
