"""Pipeline configuration models with YAML/JSON serialisation support.

Each pipeline stage is described by a typed dataclass.  The top-level
:class:`PipelineConfig` aggregates all stage configs and can be loaded from
a YAML or JSON file via :func:`load_config`.

Three built-in presets are available via :func:`default_config`:

* ``"rag"`` — Standard retrieval-augmented generation pipeline.
* ``"qa"``  — Lightweight question-answering pipeline (smaller chunks).
* ``"summarise"`` — Summarisation-focused pipeline (large chunks).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage-level enums
# ---------------------------------------------------------------------------


class IngestFormat(str, Enum):
    """Source document format handled by the ingest stage.

    Attributes:
        TEXT: Plain-text files.
        PDF: PDF documents.
        HTML: HTML pages.
        AUTO: Detect format automatically from file extension.
    """

    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    AUTO = "auto"


class ChunkMethod(str, Enum):
    """Chunking strategy for the splitting stage.

    Attributes:
        FIXED: Fixed-size character or token windows.
        RECURSIVE: Recursive splitting on separators.
        SENTENCE: Split on sentence boundaries.
        SEMANTIC: Embedding-based semantic splitting.
    """

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"


class EmbedBackend(str, Enum):
    """Embedding model backend.

    Attributes:
        MOCK: Deterministic mock embedder (for testing).
        OPENAI: OpenAI text-embedding models.
        SBERT: Sentence-Transformers / SBERT models.
    """

    MOCK = "mock"
    OPENAI = "openai"
    SBERT = "sbert"


class StoreBackend(str, Enum):
    """Vector store backend.

    Attributes:
        MEMORY: In-process in-memory store.
        FAISS: FAISS flat index.
        CHROMA: ChromaDB persistent store.
    """

    MEMORY = "memory"
    FAISS = "faiss"
    CHROMA = "chroma"


class RetrieveMethod(str, Enum):
    """Retrieval strategy.

    Attributes:
        BM25: Sparse keyword retrieval (BM25).
        VECTOR: Dense vector similarity search.
        HYBRID: Combined BM25 + vector retrieval.
        RERANKED: Retrieval followed by cross-encoder reranking.
    """

    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class GenerateBackend(str, Enum):
    """LLM backend for the generation stage.

    Attributes:
        MOCK: Deterministic mock LLM (for testing).
        OPENAI: OpenAI chat-completion models.
    """

    MOCK = "mock"
    OPENAI = "openai"


# ---------------------------------------------------------------------------
# Per-stage config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IngestConfig:
    """Configuration for the document ingest stage.

    Attributes:
        format: Expected document format (or AUTO to detect from extension).
        encoding: Character encoding to use when reading text files.
        extra: Backend-specific extra parameters.
    """

    format: IngestFormat = IngestFormat.AUTO
    encoding: str = "utf-8"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkConfig:
    """Configuration for the text chunking stage.

    Attributes:
        method: Chunking strategy to apply.
        chunk_size: Target size of each chunk (characters or tokens).
        chunk_overlap: Number of characters/tokens to overlap between chunks.
        extra: Strategy-specific extra parameters.
    """

    method: ChunkMethod = ChunkMethod.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 64
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate chunk parameters after initialisation."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )


@dataclass
class EmbedConfig:
    """Configuration for the embedding generation stage.

    Attributes:
        backend: Embedding model backend to use.
        model: Model identifier (backend-specific, e.g. ``"text-embedding-3-small"``).
        dimensions: Expected output dimension (0 = determined by the model).
        batch_size: Number of texts to embed in a single API call.
        extra: Backend-specific extra parameters.
    """

    backend: EmbedBackend = EmbedBackend.MOCK
    model: str = ""
    dimensions: int = 0
    batch_size: int = 64
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoreConfig:
    """Configuration for the vector store stage.

    Attributes:
        backend: Vector store backend to use.
        collection: Name of the collection / index.
        persist_path: Directory for on-disk persistence (ignored by MEMORY backend).
        extra: Backend-specific extra parameters.
    """

    backend: StoreBackend = StoreBackend.MEMORY
    collection: str = "lexisearch"
    persist_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieveConfig:
    """Configuration for the retrieval stage.

    Attributes:
        method: Retrieval strategy.
        top_k: Maximum number of results to return.
        score_threshold: Minimum relevance score; lower-scoring results are dropped.
        alpha: Hybrid weight in ``[0, 1]`` (1 = pure vector, 0 = pure BM25).
        extra: Retrieval-specific extra parameters.
    """

    method: RetrieveMethod = RetrieveMethod.HYBRID
    top_k: int = 5
    score_threshold: float = 0.0
    alpha: float = 0.7
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate retrieval parameters."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")


@dataclass
class GenerateConfig:
    """Configuration for the LLM generation stage.

    Attributes:
        backend: LLM provider to use.
        model: Model identifier (e.g., ``"gpt-4o-mini"``).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the completion.
        prompt_template: Name of the built-in prompt template to use.
        extra: Provider-specific extra parameters.
    """

    backend: GenerateBackend = GenerateBackend.MOCK
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    prompt_template: str = "rag_qa"
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Top-level configuration for a complete LexiSearch pipeline.

    All stage configs are optional — stages that are not provided use defaults.

    Attributes:
        name: Human-readable pipeline name.
        description: Optional free-text description.
        ingest: Ingest stage configuration.
        chunk: Chunking stage configuration.
        embed: Embedding stage configuration.
        store: Vector store stage configuration.
        retrieve: Retrieval stage configuration.
        generate: Generation stage configuration.
        extra: Pipeline-level extra parameters.
    """

    name: str = "lexisearch"
    description: str = ""
    ingest: IngestConfig = field(default_factory=IngestConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the config to a plain dictionary.

        Enum values are converted to their string representations.

        Returns:
            Nested dict representation of the config.
        """
        raw = asdict(self)
        result: dict[str, Any] = _enums_to_str(raw)
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialise the config to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            Pretty-printed JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Write the config to a JSON file.

        Args:
            path: Destination file path.  The file extension is ignored —
                JSON format is always used.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        logger.debug("Saved pipeline config to %s", p)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _enums_to_str(obj: Any) -> Any:
    """Recursively convert Enum values to their string (.value) representations.

    Args:
        obj: Arbitrary nested structure (dict, list, Enum, scalar).

    Returns:
        The same structure with Enum instances replaced by their values.
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _enums_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_enums_to_str(i) for i in obj]
    return obj


def _build_stage_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a stage config dataclass from a dictionary.

    Enum fields are resolved from their string values automatically.

    Args:
        cls: The dataclass to instantiate.
        data: Dictionary of field values.

    Returns:
        An instance of *cls* populated from *data*.
    """
    import dataclasses

    init_kwargs: dict[str, Any] = {}
    type_hints = {f.name: f.type for f in dataclasses.fields(cls)}

    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = type_hints[f.name]

        # Resolve Enum fields by looking up the annotation in module globals.
        hint_type = globals().get(hint) if isinstance(hint, str) else hint

        if hint_type is not None and isinstance(hint_type, type) and issubclass(hint_type, Enum):
            try:
                value = hint_type(value)
            except ValueError:
                logger.warning(
                    "Unknown value %r for %s.%s — keeping raw", value, cls.__name__, f.name
                )

        init_kwargs[f.name] = value

    return cls(**init_kwargs)


def from_dict(data: dict[str, Any]) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from a plain dictionary.

    Args:
        data: Nested dictionary, typically loaded from JSON or YAML.

    Returns:
        A fully populated :class:`PipelineConfig`.
    """
    stage_map: dict[str, tuple[str, type]] = {
        "ingest": ("ingest", IngestConfig),
        "chunk": ("chunk", ChunkConfig),
        "embed": ("embed", EmbedConfig),
        "store": ("store", StoreConfig),
        "retrieve": ("retrieve", RetrieveConfig),
        "generate": ("generate", GenerateConfig),
    }

    kwargs: dict[str, Any] = {}
    for key, (attr, cls) in stage_map.items():
        if key in data:
            kwargs[attr] = _build_stage_from_dict(cls, data[key])

    for top_key in ("name", "description", "extra"):
        if top_key in data:
            kwargs[top_key] = data[top_key]

    return PipelineConfig(**kwargs)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a JSON or YAML file.

    YAML support is optional: if PyYAML is not installed, only JSON files are
    supported.

    Args:
        path: Path to the configuration file.  Extension determines format:
            ``.yaml`` / ``.yml`` → YAML; everything else → JSON.

    Returns:
        A :class:`PipelineConfig` populated from the file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be parsed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix in (".yaml", ".yml"):
        try:
            yaml = import_module("yaml")
            safe_load = cast("Any", yaml).safe_load
            raw: dict[str, Any] = safe_load(text) or {}
        except ImportError as exc:
            raise ValueError(
                "PyYAML is required to load YAML configs. Install it with: pip install pyyaml"
            ) from exc
        except Exception as exc:
            raise ValueError(f"Failed to parse YAML config from {p}: {exc}") from exc
    else:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON config from {p}: {exc}") from exc

    return from_dict(raw)


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------


def default_config(preset: str = "rag") -> PipelineConfig:
    """Return a built-in preset :class:`PipelineConfig`.

    Available presets:

    * ``"rag"``      — Standard RAG pipeline (hybrid retrieval, moderate chunk size).
    * ``"qa"``       — Lightweight QA (small chunks, pure vector retrieval).
    * ``"summarise"`` — Summarisation (large chunks, BM25 retrieval).

    Args:
        preset: Name of the preset configuration.

    Returns:
        A :class:`PipelineConfig` instance pre-configured for the use case.

    Raises:
        ValueError: If *preset* is not a recognised name.
    """
    presets: dict[str, PipelineConfig] = {
        "rag": PipelineConfig(
            name="rag",
            description="Standard retrieval-augmented generation pipeline.",
            ingest=IngestConfig(format=IngestFormat.AUTO),
            chunk=ChunkConfig(method=ChunkMethod.RECURSIVE, chunk_size=512, chunk_overlap=64),
            embed=EmbedConfig(backend=EmbedBackend.MOCK, dimensions=384),
            store=StoreConfig(backend=StoreBackend.MEMORY, collection="rag"),
            retrieve=RetrieveConfig(method=RetrieveMethod.HYBRID, top_k=5, alpha=0.7),
            generate=GenerateConfig(backend=GenerateBackend.MOCK, prompt_template="rag_qa"),
        ),
        "qa": PipelineConfig(
            name="qa",
            description="Lightweight question-answering pipeline with small chunks.",
            ingest=IngestConfig(format=IngestFormat.AUTO),
            chunk=ChunkConfig(method=ChunkMethod.SENTENCE, chunk_size=256, chunk_overlap=32),
            embed=EmbedConfig(backend=EmbedBackend.MOCK, dimensions=384),
            store=StoreConfig(backend=StoreBackend.MEMORY, collection="qa"),
            retrieve=RetrieveConfig(method=RetrieveMethod.VECTOR, top_k=3, alpha=1.0),
            generate=GenerateConfig(backend=GenerateBackend.MOCK, prompt_template="rag_qa"),
        ),
        "summarise": PipelineConfig(
            name="summarise",
            description="Summarisation pipeline with large overlapping chunks.",
            ingest=IngestConfig(format=IngestFormat.AUTO),
            chunk=ChunkConfig(method=ChunkMethod.FIXED, chunk_size=1024, chunk_overlap=128),
            embed=EmbedConfig(backend=EmbedBackend.MOCK, dimensions=384),
            store=StoreConfig(backend=StoreBackend.MEMORY, collection="summarise"),
            retrieve=RetrieveConfig(method=RetrieveMethod.BM25, top_k=8, alpha=0.0),
            generate=GenerateConfig(backend=GenerateBackend.MOCK, prompt_template="rag_summarise"),
        ),
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset {preset!r}. Available: {sorted(presets)}")
    return presets[preset]
