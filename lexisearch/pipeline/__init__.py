"""LexiSearch pipeline orchestration layer.

This sub-package wires all LexiSearch stages into end-to-end pipelines:
document ingestion → chunking → embedding → vector indexing → retrieval →
LLM generation.

Quick start::

    from lexisearch.pipeline import PipelineBuilder, PipelineRunner
    from lexisearch.embeddings import MockEmbedder
    from lexisearch.generation import MockLLM
    from lexisearch.models import Document

    pipeline = (
        PipelineBuilder.create("demo")
        .embed(MockEmbedder())
        .store()
        .retrieve(top_k=3)
        .generate(MockLLM())
        .build()
    )

    runner = PipelineRunner(pipeline)
    runner.ingest_documents([Document(content="RAG pipelines combine search and generation.")])
    result = runner.query("What do RAG pipelines do?")
    print(result.answer)

Public API
----------
Builder
~~~~~~~
.. autosummary::
   :nosignatures:

   PipelineBuilder
   BuiltPipeline
   PipelineBuilderError

Runner
~~~~~~
.. autosummary::
   :nosignatures:

   PipelineRunner
   IngestResult
   QueryResult
   PipelineError

Config
~~~~~~
.. autosummary::
   :nosignatures:

   PipelineConfig
   IngestConfig
   ChunkConfig
   EmbedConfig
   StoreConfig
   RetrieveConfig
   GenerateConfig
   IngestFormat
   ChunkMethod
   EmbedBackend
   StoreBackend
   RetrieveMethod
   GenerateBackend
   load_config
   from_dict
   default_config

Registry
~~~~~~~~
.. autosummary::
   :nosignatures:

   ComponentRegistry
   ComponentKind
   ComponentInfo
   RegistryError
   discover_plugins
   registry

Events
~~~~~~
.. autosummary::
   :nosignatures:

   EventBus
   EventType
   PipelineEvent
   EventError
"""

from lexisearch.pipeline.builder import (
    BuiltPipeline,
    PipelineBuilder,
    PipelineBuilderError,
)
from lexisearch.pipeline.config import (
    ChunkConfig,
    ChunkMethod,
    EmbedBackend,
    EmbedConfig,
    GenerateBackend,
    GenerateConfig,
    IngestConfig,
    IngestFormat,
    PipelineConfig,
    RetrieveConfig,
    RetrieveMethod,
    StoreBackend,
    StoreConfig,
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
    registry,
)
from lexisearch.pipeline.runner import (
    IngestResult,
    PipelineError,
    PipelineRunner,
    QueryResult,
)

__all__ = [
    "BuiltPipeline",
    "ChunkConfig",
    "ChunkMethod",
    "ComponentInfo",
    "ComponentKind",
    "ComponentRegistry",
    "EmbedBackend",
    "EmbedConfig",
    "EventBus",
    "EventError",
    "EventType",
    "GenerateBackend",
    "GenerateConfig",
    "IngestConfig",
    "IngestFormat",
    "IngestResult",
    "PipelineBuilder",
    "PipelineBuilderError",
    "PipelineConfig",
    "PipelineError",
    "PipelineEvent",
    "PipelineRunner",
    "QueryResult",
    "RegistryError",
    "RetrieveConfig",
    "RetrieveMethod",
    "StoreBackend",
    "StoreConfig",
    "default_config",
    "discover_plugins",
    "from_dict",
    "load_config",
    "registry",
]
