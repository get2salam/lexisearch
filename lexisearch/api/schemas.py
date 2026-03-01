"""Request and response schemas for the LexiSearch REST API.

All schemas use only stdlib-compatible types so the API layer can be imported
without FastAPI/Pydantic installed (graceful degradation).  When FastAPI IS
available the schemas become proper Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Ingest schemas
# ---------------------------------------------------------------------------


@dataclass
class IngestRequest:
    """Payload for the document-ingest endpoint."""

    content: str
    """Raw text content of the document."""

    title: str = ""
    """Human-readable document title."""

    source: str = ""
    """URI or path this document came from."""

    doc_id: str = ""
    """Optional caller-supplied stable identifier."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key/value metadata stored alongside the document."""

    chunk_size: int = 512
    """Token target for each chunk (passed to the active chunker)."""

    chunk_overlap: int = 64
    """Overlap in tokens between consecutive chunks."""


@dataclass
class IngestResponse:
    """Result returned by the document-ingest endpoint."""

    doc_id: str
    """Stable ID assigned to the ingested document."""

    chunks_created: int
    """Number of chunks produced from the document."""

    tokens_estimated: int
    """Rough token count (word-based estimate)."""

    status: str = "ok"
    """``"ok"`` on success, ``"error"`` on failure."""

    message: str = ""
    """Human-readable detail (populated on error)."""


# ---------------------------------------------------------------------------
# Query schemas
# ---------------------------------------------------------------------------


@dataclass
class QueryRequest:
    """Payload for the query/search endpoint."""

    query: str
    """Natural-language question or keyword query."""

    top_k: int = 5
    """Number of chunks to retrieve."""

    filters: dict[str, Any] = field(default_factory=dict)
    """Optional metadata filters applied before scoring."""

    include_sources: bool = True
    """Whether to include source attribution in the response."""

    stream: bool = False
    """Request a streaming response (SSE)."""


@dataclass
class SourceInfo:
    """Source attribution entry returned alongside an answer."""

    chunk_id: str
    doc_id: str
    title: str
    score: float
    snippet: str


@dataclass
class QueryResponse:
    """Result returned by the query endpoint."""

    answer: str
    """Generated answer text."""

    sources: list[SourceInfo] = field(default_factory=list)
    """Retrieved chunks used to ground the answer."""

    query: str = ""
    """Echo of the original query."""

    latency_ms: float = 0.0
    """Wall-clock latency for the full pipeline pass (ms)."""

    status: str = "ok"
    message: str = ""


# ---------------------------------------------------------------------------
# Evaluation schemas
# ---------------------------------------------------------------------------


@dataclass
class EvalSampleRequest:
    """One evaluation sample submitted to the eval endpoint."""

    question: str
    contexts: list[str]
    answer: str
    reference: str = ""


@dataclass
class EvalRequest:
    """Payload for the evaluation endpoint."""

    samples: list[EvalSampleRequest]
    metrics: list[str] = field(default_factory=list)
    """Metric names to compute.  Empty list means all available metrics."""


@dataclass
class MetricScoreItem:
    """Per-metric score for one sample."""

    metric: str
    score: float
    passed: bool


@dataclass
class SampleResultItem:
    """Evaluation result for a single sample."""

    question: str
    metric_scores: list[MetricScoreItem]
    overall: float


@dataclass
class EvalResponse:
    """Aggregated evaluation report."""

    num_samples: int
    aggregate: dict[str, float]
    """Metric-name → mean score across all samples."""

    samples: list[SampleResultItem] = field(default_factory=list)
    status: str = "ok"
    message: str = ""


# ---------------------------------------------------------------------------
# Health / info schemas
# ---------------------------------------------------------------------------


@dataclass
class HealthResponse:
    """Liveness probe response."""

    status: str = "healthy"
    version: str = ""
    components: dict[str, str] = field(default_factory=dict)


@dataclass
class IndexStatsResponse:
    """Statistics about the current vector index."""

    total_documents: int = 0
    total_chunks: int = 0
    embedding_dim: int = 0
    vector_store: str = "memory"
    embedder: str = "mock"
