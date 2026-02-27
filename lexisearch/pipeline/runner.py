"""Pipeline runner: end-to-end document ingestion, indexing, and querying.

The :class:`PipelineRunner` accepts a :class:`~lexisearch.pipeline.builder.BuiltPipeline`
and coordinates the full workflow:

1. **Ingest** — Load raw content into :class:`~lexisearch.models.Document` objects.
2. **Chunk** — Split documents into :class:`~lexisearch.models.Chunk` objects.
3. **Embed** — Generate :class:`~lexisearch.models.Embedding` vectors for each chunk.
4. **Index** — Add :class:`~lexisearch.models.EmbeddedChunk` objects to the vector store.
5. **Retrieve** — Given a query, return ranked :class:`~lexisearch.models.SearchResult` list.
6. **Generate** — Produce an LLM answer using retrieved context.

Both sync and async paths are provided.  The async methods run the
compute-heavy stages (embed, index) in an :func:`asyncio.get_event_loop`
executor, keeping the event loop unblocked.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lexisearch.models import Chunk, Document, EmbeddedChunk  # noqa: TC001

if TYPE_CHECKING:
    from lexisearch.pipeline.builder import BuiltPipeline
    from lexisearch.pipeline.events import EventBus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Result of the ingest + chunk + embed + index stages.

    Attributes:
        run_id: Unique identifier for this run.
        document_count: Number of documents processed.
        chunk_count: Total chunks produced.
        embedded_count: Number of chunks successfully embedded and indexed.
        latency_ms: Total wall-clock time for all stages in milliseconds.
        stage_latencies: Per-stage latency breakdown (stage name → ms).
        errors: Any non-fatal errors encountered (document id → error message).
    """

    run_id: str
    document_count: int = 0
    chunk_count: int = 0
    embedded_count: int = 0
    latency_ms: float = 0.0
    stage_latencies: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True when all documents were processed without errors.

        Returns:
            Whether the ingest run completed without errors.
        """
        return not self.errors

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"IngestResult(run={self.run_id!r}, docs={self.document_count}, "
            f"chunks={self.chunk_count}, embedded={self.embedded_count}, "
            f"latency={self.latency_ms:.1f}ms, errors={len(self.errors)})"
        )


@dataclass
class QueryResult:
    """Result of the retrieve + generate stages.

    Attributes:
        run_id: Unique identifier for this query run.
        query: The original query string.
        answer: The LLM-generated answer.
        sources: Source chunks used to produce the answer.
        retrieval_latency_ms: Time spent in the retrieval stage.
        generation_latency_ms: Time spent in the generation stage.
        total_latency_ms: Total end-to-end latency.
        metadata: Arbitrary extra metadata (model name, token usage, etc.).
    """

    run_id: str
    query: str
    answer: str
    sources: list[Chunk] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        preview = self.answer[:60] + "..." if len(self.answer) > 60 else self.answer
        return (
            f"QueryResult(run={self.run_id!r}, query={self.query!r}, "
            f"sources={len(self.sources)}, answer={preview!r})"
        )


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

#: Signature: (step: str, current: int, total: int, data: dict) -> None
ProgressCallback = Any  # Callable[[str, int, int, dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """Raised when a pipeline stage fails fatally.

    Attributes:
        stage: Name of the failing stage.
        cause: The underlying exception.
    """

    def __init__(self, stage: str, cause: Exception) -> None:
        """Initialise a PipelineError.

        Args:
            stage: Name of the pipeline stage that failed.
            cause: The original exception.
        """
        super().__init__(f"Pipeline stage {stage!r} failed: {cause}")
        self.stage = stage
        self.cause = cause


class PipelineRunner:
    """Coordinates end-to-end pipeline execution.

    Ingest, chunk, embed, index, retrieve, and generate are orchestrated in
    order.  Each stage fires events on the optional :class:`~lexisearch.pipeline.events.EventBus`
    and reports progress via the optional *progress_callback*.

    Args:
        pipeline: A fully constructed :class:`~lexisearch.pipeline.builder.BuiltPipeline`.
        progress_callback: Optional callable invoked at each progress update.
            Signature: ``(step, current, total, data) -> None``.
        max_retries: Number of times to retry a failed LLM generation before
            re-raising (default 2).

    Examples:
        >>> from lexisearch.embeddings import MockEmbedder
        >>> from lexisearch.generation import MockLLM
        >>> from lexisearch.pipeline.builder import PipelineBuilder
        >>> pipeline = (
        ...     PipelineBuilder("test")
        ...     .embed(MockEmbedder())
        ...     .store()
        ...     .retrieve()
        ...     .generate(MockLLM())
        ...     .build()
        ... )
        >>> runner = PipelineRunner(pipeline)
        >>> doc = Document(content="Dense retrieval uses neural embeddings.")
        >>> result = runner.ingest_documents([doc])
        >>> result.document_count
        1
    """

    def __init__(
        self,
        pipeline: BuiltPipeline,
        progress_callback: ProgressCallback | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialise the runner.

        Args:
            pipeline: The assembled pipeline to run.
            progress_callback: Optional progress notification callable.
            max_retries: Retry limit for generation failures.
        """
        self._pipeline = pipeline
        self._progress_callback = progress_callback
        self._max_retries = max_retries
        self._bus: EventBus | None = pipeline.event_bus

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def pipeline_id(self) -> str:
        """Return the pipeline's unique identifier.

        Returns:
            Pipeline ID string.
        """
        return self._pipeline.pipeline_id

    def _emit(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Call *method* on the event bus if one is configured.

        Args:
            method: Name of the EventBus method to call.
            *args: Positional arguments forwarded to the method.
            **kwargs: Keyword arguments forwarded to the method.
        """
        if self._bus is not None:
            getattr(self._bus, method)(*args, **kwargs)

    def _progress(
        self, step: str, current: int, total: int, data: dict[str, Any] | None = None
    ) -> None:
        """Fire a progress callback and event.

        Args:
            step: Current pipeline step name.
            current: Items processed so far.
            total: Total items.
            data: Optional extra context.
        """
        if self._progress_callback is not None:
            try:
                self._progress_callback(step, current, total, data or {})
            except Exception as exc:
                logger.warning("Progress callback raised: %s", exc)
        self._emit("emit_progress", self.pipeline_id, step, current, total, data)

    @staticmethod
    def _new_run_id() -> str:
        """Generate a new unique run identifier.

        Returns:
            Short hex run identifier.
        """
        return uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------
    # Ingest pipeline
    # ------------------------------------------------------------------

    def ingest_documents(
        self,
        documents: list[Document],
        batch_size: int = 32,
    ) -> IngestResult:
        """Chunk, embed, and index a list of pre-loaded documents.

        Args:
            documents: Documents to process.
            batch_size: Number of chunks to embed per batch.

        Returns:
            An :class:`IngestResult` summarising the run.

        Raises:
            PipelineError: If a critical stage fails.
        """
        run_id = self._new_run_id()
        t_start = time.perf_counter()
        result = IngestResult(run_id=run_id, document_count=len(documents))

        self._emit("emit_start", self.pipeline_id, {"run_id": run_id, "documents": len(documents)})

        all_chunks: list[Chunk] = []
        t0 = time.perf_counter()
        self._emit("emit_step_start", self.pipeline_id, "chunk")
        try:
            for i, doc in enumerate(documents):
                try:
                    chunks = self._pipeline.chunker.chunk_many([doc])
                    all_chunks.extend(chunks)
                except Exception as exc:
                    logger.warning("Chunking failed for document %s: %s", doc.id, exc)
                    result.errors[doc.id] = f"chunk: {exc}"
                self._progress("chunk", i + 1, len(documents))
            result.chunk_count = len(all_chunks)
        except Exception as exc:
            self._emit("emit_error", self.pipeline_id, exc, "chunk")
            raise PipelineError("chunk", exc) from exc

        result.stage_latencies["chunk"] = (time.perf_counter() - t0) * 1000
        self._emit("emit_step_finish", self.pipeline_id, "chunk", {"chunks": result.chunk_count})

        # --- Embed ---
        t0 = time.perf_counter()
        self._emit("emit_step_start", self.pipeline_id, "embed")
        embedded: list[EmbeddedChunk] = []
        try:
            for batch_start in range(0, len(all_chunks), batch_size):
                batch = all_chunks[batch_start : batch_start + batch_size]
                batch_embedded = self._pipeline.embedder.embed_chunks(batch)
                embedded.extend(batch_embedded)
                self._progress("embed", batch_start + len(batch), len(all_chunks))
            result.embedded_count = len(embedded)
        except Exception as exc:
            self._emit("emit_error", self.pipeline_id, exc, "embed")
            raise PipelineError("embed", exc) from exc

        result.stage_latencies["embed"] = (time.perf_counter() - t0) * 1000
        self._emit(
            "emit_step_finish", self.pipeline_id, "embed", {"embedded": result.embedded_count}
        )

        # --- Index ---
        t0 = time.perf_counter()
        self._emit("emit_step_start", self.pipeline_id, "index")
        try:
            self._pipeline.store.add(embedded)
        except Exception as exc:
            self._emit("emit_error", self.pipeline_id, exc, "index")
            raise PipelineError("index", exc) from exc

        result.stage_latencies["index"] = (time.perf_counter() - t0) * 1000
        self._emit(
            "emit_step_finish", self.pipeline_id, "index", {"indexed": result.embedded_count}
        )

        result.latency_ms = (time.perf_counter() - t_start) * 1000
        self._emit(
            "emit_finish",
            self.pipeline_id,
            {"run_id": run_id, "latency_ms": result.latency_ms},
        )
        return result

    def ingest_from_loader(
        self,
        path: str,
        batch_size: int = 32,
    ) -> IngestResult:
        """Load documents from *path* using the configured loader, then ingest.

        Args:
            path: File system path to load.
            batch_size: Embedding batch size.

        Returns:
            An :class:`IngestResult` summarising the run.

        Raises:
            PipelineError: If no loader is configured or loading fails.
        """
        if self._pipeline.loader is None:
            raise PipelineError(
                "ingest",
                ValueError("No loader configured. Call .ingest(loader=...) on the builder."),
            )
        try:
            documents = self._pipeline.loader.load(path)
        except Exception as exc:
            raise PipelineError("ingest", exc) from exc

        return self.ingest_documents(documents, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Query pipeline
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        """Retrieve relevant chunks and generate an answer.

        Args:
            question: The natural-language query.
            top_k: Override the configured ``top_k`` (optional).

        Returns:
            A :class:`QueryResult` with the generated answer and sources.

        Raises:
            PipelineError: If retrieval or generation fails fatally.
        """
        run_id = self._new_run_id()
        t_total = time.perf_counter()

        # --- Retrieve ---
        t0 = time.perf_counter()
        self._emit("emit_step_start", self.pipeline_id, "retrieve", {"query": question})
        try:
            search_resp = self._pipeline.retriever.search(question, top_k=top_k)
        except Exception as exc:
            self._emit("emit_error", self.pipeline_id, exc, "retrieve")
            raise PipelineError("retrieve", exc) from exc
        retrieval_ms = (time.perf_counter() - t0) * 1000
        self._emit(
            "emit_step_finish",
            self.pipeline_id,
            "retrieve",
            {"results": len(search_resp.results)},
        )

        sources = [r.chunk for r in search_resp.results]

        # --- Generate ---
        t0 = time.perf_counter()
        self._emit("emit_step_start", self.pipeline_id, "generate")
        answer, gen_metadata = self._generate_with_retry(question, search_resp.results)
        generation_ms = (time.perf_counter() - t0) * 1000
        self._emit("emit_step_finish", self.pipeline_id, "generate", {"answer_len": len(answer)})

        total_ms = (time.perf_counter() - t_total) * 1000

        return QueryResult(
            run_id=run_id,
            query=question,
            answer=answer,
            sources=sources,
            retrieval_latency_ms=round(retrieval_ms, 2),
            generation_latency_ms=round(generation_ms, 2),
            total_latency_ms=round(total_ms, 2),
            metadata=gen_metadata,
        )

    def _generate_with_retry(
        self,
        question: str,
        results: list[Any],
    ) -> tuple[str, dict[str, Any]]:
        """Run generation with retry logic.

        Args:
            question: The user question.
            results: Retrieved :class:`~lexisearch.models.SearchResult` list.

        Returns:
            Tuple of (answer text, metadata dict).

        Raises:
            PipelineError: After exhausting all retries.
        """
        from lexisearch.generation import GenerationConfig, GenerationRequest, Message, MessageRole

        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[Source {i}]\n{r.chunk.content}")
        context = "\n\n".join(context_parts) if context_parts else "No relevant sources found."

        prompt = (
            f"You are a helpful assistant. Answer the question based on the context below.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        gen_cfg = self._pipeline.config.generate
        request = GenerationRequest(
            messages=[Message(role=MessageRole.USER, content=prompt)],
            config=GenerationConfig(
                temperature=gen_cfg.temperature,
                max_tokens=gen_cfg.max_tokens,
            ),
        )

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._pipeline.llm.complete(request)
                metadata: dict[str, Any] = {
                    "model": response.model,
                    "finish_reason": response.finish_reason.value,
                    "latency_ms": response.latency_ms,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
                return response.content, metadata
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    logger.warning("Generation attempt %d failed: %s — retrying", attempt + 1, exc)
                else:
                    logger.error("Generation failed after %d attempts: %s", attempt + 1, exc)

        raise PipelineError("generate", last_exc or RuntimeError("Unknown generation error"))

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def aingest_documents(
        self,
        documents: list[Document],
        batch_size: int = 32,
    ) -> IngestResult:
        """Async variant of :meth:`ingest_documents`.

        Runs the blocking ingest in a thread pool executor to avoid blocking
        the event loop.

        Args:
            documents: Documents to process.
            batch_size: Embedding batch size.

        Returns:
            An :class:`IngestResult` summarising the run.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.ingest_documents(documents, batch_size=batch_size)
        )

    async def aquery(self, question: str, top_k: int | None = None) -> QueryResult:
        """Async variant of :meth:`query`.

        Runs the blocking query in a thread pool executor.

        Args:
            question: The natural-language query.
            top_k: Override the configured top_k.

        Returns:
            A :class:`QueryResult`.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.query(question, top_k=top_k))

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"PipelineRunner(pipeline={self._pipeline.pipeline_id!r}, "
            f"max_retries={self._max_retries})"
        )
