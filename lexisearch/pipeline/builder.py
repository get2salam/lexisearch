"""Fluent builder for assembling LexiSearch pipelines.

The :class:`PipelineBuilder` implements a step-by-step, method-chaining API
for composing the six pipeline stages.  Call :meth:`~PipelineBuilder.build`
at the end to obtain a validated :class:`BuiltPipeline` ready for execution
by :class:`~lexisearch.pipeline.runner.PipelineRunner`.

Usage
-----
::

    from lexisearch.pipeline.builder import PipelineBuilder
    from lexisearch.embeddings import MockEmbedder
    from lexisearch.generation import MockLLM

    pipeline = (
        PipelineBuilder("my-pipeline")
        .chunk(chunk_size=256)
        .embed(MockEmbedder())
        .store()
        .retrieve(top_k=3)
        .generate(MockLLM())
        .build()
    )

Stages
------
* :meth:`ingest`   — configure the document loader.
* :meth:`chunk`    — configure the text chunker.
* :meth:`embed`    — set the embedding model.
* :meth:`store`    — set the vector store (default: in-memory).
* :meth:`retrieve` — configure the retriever.
* :meth:`generate` — set the language model.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lexisearch.embeddings.base import BaseEmbedder
    from lexisearch.generation.base import BaseLLM
    from lexisearch.ingest.base import BaseLoader
    from lexisearch.pipeline.events import EventBus
    from lexisearch.retrieval.base import BaseRetriever
    from lexisearch.vectorstore.base import BaseVectorStore

from lexisearch.pipeline.config import (
    ChunkConfig,
    ChunkMethod,
    EmbedConfig,
    GenerateConfig,
    IngestConfig,
    PipelineConfig,
    RetrieveConfig,
    StoreConfig,
)


class PipelineBuilderError(Exception):
    """Raised when a :class:`PipelineBuilder` is mis-configured.

    Attributes:
        missing_stages: Names of required stages that were not configured.
    """

    def __init__(self, message: str, missing_stages: list[str] | None = None) -> None:
        """Initialise a PipelineBuilderError.

        Args:
            message: Human-readable error description.
            missing_stages: Names of stages that are not configured.
        """
        super().__init__(message)
        self.missing_stages = missing_stages or []


@dataclass
class BuiltPipeline:
    """A fully validated, executable pipeline produced by :class:`PipelineBuilder`.

    Attributes:
        pipeline_id: Unique identifier for this pipeline instance.
        config: The aggregate :class:`~lexisearch.pipeline.config.PipelineConfig`.
        loader: Optional document loader component.
        chunker: Optional chunker component.
        embedder: Embedding model component.
        store: Vector store component.
        retriever: Retriever component.
        llm: Language model component.
        event_bus: Optional event bus for lifecycle notifications.
        metadata: Arbitrary caller-supplied metadata.
    """

    pipeline_id: str
    config: PipelineConfig
    loader: Any  # BaseLoader | None
    chunker: Any  # BaseChunker
    embedder: Any  # BaseEmbedder
    store: Any  # BaseVectorStore
    retriever: Any  # BaseRetriever
    llm: Any  # BaseLLM
    event_bus: Any = None  # EventBus | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"BuiltPipeline(id={self.pipeline_id!r}, "
            f"embedder={type(self.embedder).__name__!r}, "
            f"store={type(self.store).__name__!r}, "
            f"retriever={type(self.retriever).__name__!r}, "
            f"llm={type(self.llm).__name__!r})"
        )


class PipelineBuilder:
    """Fluent builder for constructing :class:`BuiltPipeline` instances.

    Each stage method returns ``self``, enabling method chaining.

    The minimum required stages are:

    * :meth:`embed` — must be called with an embedder instance.
    * :meth:`store` — vector store (default in-memory if not called).
    * :meth:`retrieve` — retriever (default vector retriever if not called).
    * :meth:`generate` — must be called with an LLM instance.

    :meth:`ingest` and :meth:`chunk` are optional (documents can be provided
    directly to the runner).

    Args:
        name: Human-readable pipeline name (also used as pipeline_id prefix).
        pipeline_id: Explicit pipeline identifier.  Auto-generated if omitted.

    Examples:
        >>> from lexisearch.generation import MockLLM
        >>> from lexisearch.embeddings import MockEmbedder
        >>> pipeline = (
        ...     PipelineBuilder("test")
        ...     .embed(MockEmbedder())
        ...     .store()
        ...     .retrieve()
        ...     .generate(MockLLM())
        ...     .build()
        ... )
        >>> pipeline.pipeline_id.startswith("test-")
        True
    """

    # Required stages — must all be non-None before build() succeeds.
    _REQUIRED: tuple[str, ...] = ("embedder", "llm")

    def __init__(self, name: str = "pipeline", pipeline_id: str = "") -> None:
        """Initialise the builder.

        Args:
            name: Pipeline name (used as prefix for auto-generated IDs).
            pipeline_id: Explicit ID; auto-generated when empty.
        """
        self._name = name
        self._pipeline_id = pipeline_id or f"{name}-{uuid.uuid4().hex[:8]}"

        # Stage components
        self._loader: Any = None
        self._chunker: Any = None
        self._embedder: Any = None
        self._store: Any = None
        self._retriever: Any = None
        self._llm: Any = None
        self._event_bus: Any = None

        # Stage configs (updated by each fluent method)
        self._ingest_cfg = IngestConfig()
        self._chunk_cfg = ChunkConfig()
        self._embed_cfg = EmbedConfig()
        self._store_cfg = StoreConfig()
        self._retrieve_cfg = RetrieveConfig()
        self._generate_cfg = GenerateConfig()

        self._metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Stage configuration methods
    # ------------------------------------------------------------------

    def ingest(
        self,
        loader: BaseLoader | None = None,
        *,
        encoding: str = "utf-8",
        **extra: Any,
    ) -> PipelineBuilder:
        """Configure the document ingest stage.

        Args:
            loader: A concrete :class:`~lexisearch.ingest.base.BaseLoader`
                instance.  When ``None``, the builder will attempt to
                auto-select a loader based on the document format.
            encoding: Character encoding for text files.
            **extra: Extra parameters stored in :class:`IngestConfig`.

        Returns:
            The builder instance (for chaining).
        """
        self._loader = loader
        self._ingest_cfg = IngestConfig(encoding=encoding, extra=dict(extra))
        return self

    def chunk(
        self,
        chunker: Any = None,
        *,
        method: ChunkMethod = ChunkMethod.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        **extra: Any,
    ) -> PipelineBuilder:
        """Configure the text chunking stage.

        Args:
            chunker: A concrete chunker instance.  When ``None``, one is
                created from *method*, *chunk_size*, and *chunk_overlap*.
            method: Chunking strategy (used only when *chunker* is None).
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.
            **extra: Extra parameters stored in :class:`ChunkConfig`.

        Returns:
            The builder instance (for chaining).
        """
        self._chunker = chunker
        self._chunk_cfg = ChunkConfig(
            method=method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extra=dict(extra),
        )
        return self

    def embed(
        self,
        embedder: BaseEmbedder | None = None,
        *,
        dimensions: int = 0,
        batch_size: int = 64,
        **extra: Any,
    ) -> PipelineBuilder:
        """Configure the embedding stage.

        Args:
            embedder: A concrete :class:`~lexisearch.embeddings.base.BaseEmbedder`
                instance.  **Required** — the builder will raise if this is
                None at :meth:`build` time.
            dimensions: Expected embedding dimension (informational).
            batch_size: Embedding batch size.
            **extra: Extra parameters stored in :class:`EmbedConfig`.

        Returns:
            The builder instance (for chaining).
        """
        self._embedder = embedder
        self._embed_cfg = EmbedConfig(
            dimensions=dimensions,
            batch_size=batch_size,
            extra=dict(extra),
        )
        return self

    def store(
        self,
        vector_store: BaseVectorStore | None = None,
        *,
        collection: str = "lexisearch",
        persist_path: str = "",
        **extra: Any,
    ) -> PipelineBuilder:
        """Configure the vector store stage.

        Args:
            vector_store: A concrete
                :class:`~lexisearch.vectorstore.base.BaseVectorStore`
                instance.  When ``None``, an in-memory store is created
                automatically.
            collection: Collection name for the store.
            persist_path: On-disk persistence path (empty = no persistence).
            **extra: Extra parameters stored in :class:`StoreConfig`.

        Returns:
            The builder instance (for chaining).
        """
        self._store = vector_store
        self._store_cfg = StoreConfig(
            collection=collection,
            persist_path=persist_path,
            extra=dict(extra),
        )
        return self

    def retrieve(
        self,
        retriever: BaseRetriever | None = None,
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        alpha: float = 0.7,
        **extra: Any,
    ) -> PipelineBuilder:
        """Configure the retrieval stage.

        Args:
            retriever: A concrete
                :class:`~lexisearch.retrieval.base.BaseRetriever`
                instance.  When ``None``, a default vector retriever
                is constructed at :meth:`build` time.
            top_k: Number of top results to return.
            score_threshold: Minimum relevance score.
            alpha: Hybrid search weight (1 = full vector, 0 = full BM25).
            **extra: Extra parameters stored in :class:`RetrieveConfig`.

        Returns:
            The builder instance (for chaining).
        """
        self._retriever = retriever
        self._retrieve_cfg = RetrieveConfig(
            top_k=top_k,
            score_threshold=score_threshold,
            alpha=alpha,
            extra=dict(extra),
        )
        return self

    def generate(
        self,
        llm: BaseLLM | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        prompt_template: str = "rag_qa",
        **extra: Any,
    ) -> PipelineBuilder:
        """Configure the generation stage.

        Args:
            llm: A concrete :class:`~lexisearch.generation.base.BaseLLM`
                instance.  **Required** — the builder will raise if this is
                None at :meth:`build` time.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.
            prompt_template: Name of the built-in prompt template.
            **extra: Extra parameters stored in :class:`GenerateConfig`.

        Returns:
            The builder instance (for chaining).
        """
        self._llm = llm
        self._generate_cfg = GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_template=prompt_template,
            extra=dict(extra),
        )
        return self

    def with_events(self, event_bus: EventBus) -> PipelineBuilder:
        """Attach an :class:`~lexisearch.pipeline.events.EventBus` to this pipeline.

        Args:
            event_bus: The bus to use for lifecycle events.

        Returns:
            The builder instance (for chaining).
        """
        self._event_bus = event_bus
        return self

    def with_metadata(self, **metadata: Any) -> PipelineBuilder:
        """Attach arbitrary metadata to the pipeline.

        Args:
            **metadata: Key-value pairs stored on the :class:`BuiltPipeline`.

        Returns:
            The builder instance (for chaining).
        """
        self._metadata.update(metadata)
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Check the builder state and return a list of validation errors.

        Returns:
            A (possibly empty) list of human-readable error strings.
        """
        errors: list[str] = []

        if self._embedder is None:
            errors.append("embed() must be called with a non-None embedder instance")
        if self._llm is None:
            errors.append("generate() must be called with a non-None llm instance")

        return errors

    def build(self) -> BuiltPipeline:
        """Validate the builder state and return a :class:`BuiltPipeline`.

        Default components are created for optional stages that were not
        explicitly configured:

        * **Chunker** — :class:`~lexisearch.chunking.RecursiveChunker` with
          settings from :attr:`_chunk_cfg`.
        * **Store** — :class:`~lexisearch.vectorstore.InMemoryVectorStore`
          with ``dimensions`` taken from the embedder.
        * **Retriever** — :class:`~lexisearch.retrieval.VectorRetriever`
          backed by the store.

        Returns:
            A validated :class:`BuiltPipeline`.

        Raises:
            PipelineBuilderError: If any required stage is missing.
        """
        errors = self.validate()
        if errors:
            raise PipelineBuilderError(
                f"Pipeline validation failed: {'; '.join(errors)}",
                missing_stages=[e.split("(")[0].strip() for e in errors],
            )

        chunker = self._chunker or self._default_chunker()
        store = self._store or self._default_store()
        retriever = self._retriever or self._default_retriever(store)

        config = PipelineConfig(
            name=self._name,
            ingest=self._ingest_cfg,
            chunk=self._chunk_cfg,
            embed=self._embed_cfg,
            store=self._store_cfg,
            retrieve=self._retrieve_cfg,
            generate=self._generate_cfg,
        )

        return BuiltPipeline(
            pipeline_id=self._pipeline_id,
            config=config,
            loader=self._loader,
            chunker=chunker,
            embedder=self._embedder,
            store=store,
            retriever=retriever,
            llm=self._llm,
            event_bus=self._event_bus,
            metadata=dict(self._metadata),
        )

    # ------------------------------------------------------------------
    # Default component factories
    # ------------------------------------------------------------------

    def _default_chunker(self) -> Any:
        """Create the default chunker from :attr:`_chunk_cfg`."""
        method_map = {
            ChunkMethod.FIXED: "FixedSizeChunker",
            ChunkMethod.RECURSIVE: "RecursiveChunker",
            ChunkMethod.SENTENCE: "SentenceChunker",
            ChunkMethod.SEMANTIC: "SemanticChunker",
        }
        cls_name = method_map.get(self._chunk_cfg.method, "RecursiveChunker")
        import lexisearch.chunking as _chunking

        cls = getattr(_chunking, cls_name)
        return cls(
            chunk_size=self._chunk_cfg.chunk_size,
            chunk_overlap=self._chunk_cfg.chunk_overlap,
        )

    def _default_store(self) -> Any:
        """Create the default in-memory vector store."""
        from lexisearch.vectorstore import InMemoryVectorStore, VectorStoreConfig

        dims = getattr(self._embedder, "dimensions", lambda: 384)()
        cfg = VectorStoreConfig(
            collection_name=self._store_cfg.collection,
            dimensions=dims,
        )
        store = InMemoryVectorStore(config=cfg)
        store.initialize()
        return store

    def _default_retriever(self, store: Any) -> Any:
        """Create the default vector retriever backed by *store*."""
        from lexisearch.retrieval import VectorRetriever
        from lexisearch.retrieval.vector_retriever import VectorRetrieverConfig

        cfg = VectorRetrieverConfig(
            top_k=self._retrieve_cfg.top_k,
            score_threshold=self._retrieve_cfg.score_threshold,
        )
        return VectorRetriever(store, self._embedder, config=cfg)

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, name: str = "pipeline") -> PipelineBuilder:
        """Entry point for the fluent builder API.

        Args:
            name: Human-readable pipeline name.

        Returns:
            A new :class:`PipelineBuilder` instance.

        Examples:
            >>> from lexisearch.pipeline.builder import PipelineBuilder
            >>> builder = PipelineBuilder.create("my-pipeline")
            >>> isinstance(builder, PipelineBuilder)
            True
        """
        return cls(name=name)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        configured = []
        if self._loader:
            configured.append("ingest")
        if self._chunker:
            configured.append("chunk")
        if self._embedder:
            configured.append("embed")
        if self._store:
            configured.append("store")
        if self._retriever:
            configured.append("retrieve")
        if self._llm:
            configured.append("generate")
        return f"PipelineBuilder(id={self._pipeline_id!r}, stages={configured})"
