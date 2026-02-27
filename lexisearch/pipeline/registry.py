"""Pipeline component registry and plugin discovery.

The :class:`ComponentRegistry` maps string aliases to concrete component
classes or factory callables.  Built-in components are pre-registered; custom
components can be added via :meth:`~ComponentRegistry.register`.

Component kinds
---------------
Each pipeline stage has a ``ComponentKind``:

* ``LOADER``   — document loaders (ingest stage)
* ``CHUNKER``  — text chunkers (chunk stage)
* ``EMBEDDER`` — embedding models (embed stage)
* ``STORE``    — vector stores (store stage)
* ``RETRIEVER`` — retrievers (retrieve stage)
* ``LLM``      — language models (generate stage)

Discovery
---------
:func:`discover_plugins` scans an optional entry-point group
(``lexisearch.plugins``) if *importlib.metadata* is available, then falls
back to scanning a local ``plugins/`` directory.  Found modules are imported
and any :class:`ComponentRegistry` instances named ``registry`` at module
level are merged into the main registry.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ComponentKind(str, Enum):
    """Kind of pipeline component stored in the registry.

    Attributes:
        LOADER: Document loader for the ingest stage.
        CHUNKER: Text chunker for the splitting stage.
        EMBEDDER: Embedding model for the embedding stage.
        STORE: Vector store for the indexing stage.
        RETRIEVER: Retriever for the search stage.
        LLM: Language model for the generation stage.
    """

    LOADER = "loader"
    CHUNKER = "chunker"
    EMBEDDER = "embedder"
    STORE = "store"
    RETRIEVER = "retriever"
    LLM = "llm"


@dataclass(frozen=True)
class ComponentInfo:
    """Metadata for a registered component.

    Attributes:
        kind: The stage this component belongs to.
        alias: The short name used to look up this component.
        factory: Callable (class or function) that returns a component instance.
        description: Human-readable description.
        requires: Optional dependency specifier (e.g., ``"openai>=1.0"``).
    """

    kind: ComponentKind
    alias: str
    factory: Any  # Callable[..., Any]
    description: str = ""
    requires: str = ""


class RegistryError(Exception):
    """Raised for registry lookup or registration errors.

    Attributes:
        alias: The alias that caused the error.
        kind: The component kind, if known.
    """

    def __init__(self, message: str, alias: str = "", kind: ComponentKind | None = None) -> None:
        """Initialise a RegistryError.

        Args:
            message: Human-readable error description.
            alias: The alias that caused the error.
            kind: Component kind, if applicable.
        """
        super().__init__(message)
        self.alias = alias
        self.kind = kind


class ComponentRegistry:
    """Central registry mapping string aliases to component factories.

    All built-in LexiSearch components are pre-registered in the default
    instance (see :data:`registry`).  Additional components can be added with
    :meth:`register` or :meth:`register_many`.

    Args:
        auto_register_builtins: When ``True`` (default), pre-populate the
            registry with the standard built-in components.

    Examples:
        >>> reg = ComponentRegistry(auto_register_builtins=False)
        >>> reg.register(ComponentKind.LLM, "echo", lambda: None, "Echo LLM")
        >>> reg.get(ComponentKind.LLM, "echo") is not None
        True
    """

    def __init__(self, auto_register_builtins: bool = True) -> None:
        """Initialise the component registry.

        Args:
            auto_register_builtins: Pre-register built-in components.
        """
        self._store: dict[ComponentKind, dict[str, ComponentInfo]] = {
            kind: {} for kind in ComponentKind
        }
        if auto_register_builtins:
            self._register_builtins()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        kind: ComponentKind,
        alias: str,
        factory: Any,
        description: str = "",
        requires: str = "",
        overwrite: bool = False,
    ) -> None:
        """Register a component factory under *alias*.

        Args:
            kind: The component kind / pipeline stage.
            alias: Short name used to retrieve the component.
            factory: Class or callable that creates the component.
            description: Optional human-readable description.
            requires: Optional pip dependency string (informational only).
            overwrite: Allow replacing an existing alias.

        Raises:
            RegistryError: If *alias* already exists and *overwrite* is False.
        """
        existing = self._store[kind].get(alias)
        if existing is not None and not overwrite:
            raise RegistryError(
                f"Component {alias!r} ({kind.value}) is already registered. "
                "Use overwrite=True to replace it.",
                alias=alias,
                kind=kind,
            )
        self._store[kind][alias] = ComponentInfo(
            kind=kind,
            alias=alias,
            factory=factory,
            description=description,
            requires=requires,
        )
        logger.debug("Registered %s/%s", kind.value, alias)

    def register_many(self, components: list[ComponentInfo], overwrite: bool = False) -> None:
        """Batch-register a list of :class:`ComponentInfo` objects.

        Args:
            components: Components to register.
            overwrite: Passed to each :meth:`register` call.
        """
        for info in components:
            self.register(
                kind=info.kind,
                alias=info.alias,
                factory=info.factory,
                description=info.description,
                requires=info.requires,
                overwrite=overwrite,
            )

    def unregister(self, kind: ComponentKind, alias: str) -> bool:
        """Remove a registered component.

        Args:
            kind: The component kind.
            alias: The alias to remove.

        Returns:
            ``True`` if the alias was found and removed.
        """
        if alias in self._store[kind]:
            del self._store[kind][alias]
            logger.debug("Unregistered %s/%s", kind.value, alias)
            return True
        return False

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, kind: ComponentKind, alias: str) -> ComponentInfo:
        """Retrieve a :class:`ComponentInfo` by kind and alias.

        Args:
            kind: The component kind.
            alias: The registered alias.

        Returns:
            The :class:`ComponentInfo` for the requested component.

        Raises:
            RegistryError: If the alias is not found for this kind.
        """
        info = self._store[kind].get(alias)
        if info is None:
            available = sorted(self._store[kind])
            raise RegistryError(
                f"No {kind.value} registered as {alias!r}. Available: {available}",
                alias=alias,
                kind=kind,
            )
        return info

    def get_factory(self, kind: ComponentKind, alias: str) -> Any:
        """Return the factory callable for the given kind and alias.

        Args:
            kind: The component kind.
            alias: The registered alias.

        Returns:
            The factory callable.

        Raises:
            RegistryError: If the alias is not found.
        """
        return self.get(kind, alias).factory

    def build(self, kind: ComponentKind, alias: str, **kwargs: Any) -> Any:
        """Instantiate a component by calling its factory with *kwargs*.

        Args:
            kind: The component kind.
            alias: The registered alias.
            **kwargs: Arguments forwarded to the factory.

        Returns:
            The constructed component instance.

        Raises:
            RegistryError: If the alias is not found.
            Exception: Any exception raised by the factory.
        """
        factory = self.get_factory(kind, alias)
        return factory(**kwargs)

    def list_aliases(self, kind: ComponentKind) -> list[str]:
        """Return sorted alias names for a given component kind.

        Args:
            kind: The component kind to list.

        Returns:
            Sorted list of alias strings.
        """
        return sorted(self._store[kind])

    def list_all(self) -> dict[str, list[str]]:
        """Return a dict of all registered components grouped by kind.

        Returns:
            Dict mapping kind name → sorted list of aliases.
        """
        return {kind.value: self.list_aliases(kind) for kind in ComponentKind}

    def has(self, kind: ComponentKind, alias: str) -> bool:
        """Check whether an alias is registered for a given kind.

        Args:
            kind: The component kind.
            alias: The alias to check.

        Returns:
            ``True`` if the alias is registered.
        """
        return alias in self._store[kind]

    # ------------------------------------------------------------------
    # Built-in component registration
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        """Pre-populate the registry with built-in LexiSearch components.

        Components are registered lazily by alias; the actual import happens
        only when :meth:`build` is called.  This avoids hard import-time
        dependencies on optional packages.
        """
        self._register_loaders()
        self._register_chunkers()
        self._register_embedders()
        self._register_stores()
        self._register_retrievers()
        self._register_llms()

    def _register_loaders(self) -> None:
        """Register built-in document loaders."""

        def _text_loader_factory(**kw: Any) -> Any:
            from lexisearch.ingest import TextLoader

            return TextLoader(**kw)

        def _pdf_loader_factory(**kw: Any) -> Any:
            from lexisearch.ingest import PDFLoader

            return PDFLoader(**kw)

        def _html_loader_factory(**kw: Any) -> Any:
            from lexisearch.ingest import HTMLLoader

            return HTMLLoader(**kw)

        self.register(ComponentKind.LOADER, "text", _text_loader_factory, "Plain-text file loader")
        self.register(
            ComponentKind.LOADER, "pdf", _pdf_loader_factory, "PDF loader", requires="pymupdf"
        )
        self.register(
            ComponentKind.LOADER,
            "html",
            _html_loader_factory,
            "HTML loader",
            requires="beautifulsoup4",
        )

    def _register_chunkers(self) -> None:
        """Register built-in text chunkers."""

        def _fixed(**kw: Any) -> Any:
            from lexisearch.chunking import FixedSizeChunker

            return FixedSizeChunker(**kw)

        def _recursive(**kw: Any) -> Any:
            from lexisearch.chunking import RecursiveChunker

            return RecursiveChunker(**kw)

        def _sentence(**kw: Any) -> Any:
            from lexisearch.chunking import SentenceChunker

            return SentenceChunker(**kw)

        def _semantic(**kw: Any) -> Any:
            from lexisearch.chunking import SemanticChunker

            return SemanticChunker(**kw)

        self.register(ComponentKind.CHUNKER, "fixed", _fixed, "Fixed-size chunker")
        self.register(ComponentKind.CHUNKER, "recursive", _recursive, "Recursive chunker")
        self.register(ComponentKind.CHUNKER, "sentence", _sentence, "Sentence-boundary chunker")
        self.register(ComponentKind.CHUNKER, "semantic", _semantic, "Semantic chunker")

    def _register_embedders(self) -> None:
        """Register built-in embedding backends."""

        def _mock(**kw: Any) -> Any:
            from lexisearch.embeddings import MockEmbedder

            return MockEmbedder(**kw)

        def _openai(**kw: Any) -> Any:
            from lexisearch.embeddings import OpenAIEmbedder

            return OpenAIEmbedder(**kw)

        def _sbert(**kw: Any) -> Any:
            from lexisearch.embeddings import SentenceTransformerEmbedder

            return SentenceTransformerEmbedder(**kw)

        self.register(ComponentKind.EMBEDDER, "mock", _mock, "Deterministic mock embedder")
        self.register(
            ComponentKind.EMBEDDER,
            "openai",
            _openai,
            "OpenAI embedding API",
            requires="openai>=1.0",
        )
        self.register(
            ComponentKind.EMBEDDER,
            "sbert",
            _sbert,
            "Sentence-Transformers embedder",
            requires="sentence-transformers",
        )

    def _register_stores(self) -> None:
        """Register built-in vector stores."""

        def _memory(**kw: Any) -> Any:
            from lexisearch.vectorstore import InMemoryVectorStore

            return InMemoryVectorStore(**kw)

        def _faiss(**kw: Any) -> Any:
            from lexisearch.vectorstore import FAISSVectorStore

            return FAISSVectorStore(**kw)

        def _chroma(**kw: Any) -> Any:
            from lexisearch.vectorstore import ChromaVectorStore

            return ChromaVectorStore(**kw)

        self.register(ComponentKind.STORE, "memory", _memory, "In-memory vector store")
        self.register(
            ComponentKind.STORE, "faiss", _faiss, "FAISS flat-index store", requires="faiss-cpu"
        )
        self.register(ComponentKind.STORE, "chroma", _chroma, "ChromaDB store", requires="chromadb")

    def _register_retrievers(self) -> None:
        """Register built-in retrieval backends."""

        def _bm25(**kw: Any) -> Any:
            from lexisearch.retrieval import BM25Retriever

            return BM25Retriever(**kw)

        def _vector(**kw: Any) -> Any:
            from lexisearch.retrieval import VectorRetriever

            return VectorRetriever(**kw)

        def _hybrid(**kw: Any) -> Any:
            from lexisearch.retrieval import HybridRetriever

            return HybridRetriever(**kw)

        def _reranked(**kw: Any) -> Any:
            from lexisearch.retrieval import RerankedRetriever

            return RerankedRetriever(**kw)

        self.register(ComponentKind.RETRIEVER, "bm25", _bm25, "BM25 sparse retriever")
        self.register(ComponentKind.RETRIEVER, "vector", _vector, "Dense vector retriever")
        self.register(ComponentKind.RETRIEVER, "hybrid", _hybrid, "Hybrid BM25 + vector retriever")
        self.register(
            ComponentKind.RETRIEVER, "reranked", _reranked, "Reranked retriever (cross-encoder)"
        )

    def _register_llms(self) -> None:
        """Register built-in LLM adapters."""

        def _mock(**kw: Any) -> Any:
            from lexisearch.generation import MockLLM

            return MockLLM(**kw)

        def _openai(**kw: Any) -> Any:
            from lexisearch.generation import OpenAILLM

            return OpenAILLM(**kw)

        self.register(ComponentKind.LLM, "mock", _mock, "Deterministic mock LLM")
        self.register(
            ComponentKind.LLM,
            "openai",
            _openai,
            "OpenAI chat-completion LLM",
            requires="openai>=1.0",
        )

    def __repr__(self) -> str:
        """Return a concise string representation."""
        totals = {k.value: len(v) for k, v in self._store.items()}
        return f"ComponentRegistry({totals})"


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------


def discover_plugins(
    registry: ComponentRegistry,
    entry_point_group: str = "lexisearch.plugins",
    plugins_dir: str | None = None,
) -> int:
    """Discover and load external plugin components.

    Plugins can be distributed as Python packages that declare an entry point
    in the ``lexisearch.plugins`` group, or as standalone ``.py`` files in a
    local directory.

    Each plugin module should expose a module-level :class:`ComponentRegistry`
    named ``registry``; its contents are merged into *registry*.

    Args:
        registry: The registry to merge discovered components into.
        entry_point_group: The ``importlib.metadata`` entry-point group name.
        plugins_dir: Optional path to a directory of plugin ``.py`` files.

    Returns:
        Total number of component registrations added.
    """
    added = 0

    # 1. Entry-point discovery
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group=entry_point_group)
        for ep in eps:
            try:
                module = ep.load()
                plugin_registry: ComponentRegistry | None = getattr(module, "registry", None)
                if isinstance(plugin_registry, ComponentRegistry):
                    for kind in ComponentKind:
                        for alias, info in plugin_registry._store[kind].items():
                            if not registry.has(kind, alias):
                                registry.register(
                                    kind,
                                    alias,
                                    info.factory,
                                    info.description,
                                    info.requires,
                                )
                                added += 1
                    logger.info("Loaded plugin %r (%d components)", ep.name, added)
            except Exception as exc:
                logger.warning("Failed to load plugin %r: %s", ep.name, exc)
    except Exception as exc:
        logger.debug("Entry-point discovery unavailable: %s", exc)

    # 2. Local plugins directory
    if plugins_dir is not None:
        from pathlib import Path

        plugin_path = Path(plugins_dir)
        if plugin_path.is_dir():
            for py_file in sorted(plugin_path.glob("*.py")):
                try:
                    spec = importlib.util.spec_from_file_location(py_file.stem, py_file)  # type: ignore[attr-defined]
                    if spec is None or spec.loader is None:
                        continue
                    module = importlib.util.module_from_spec(spec)  # type: ignore[attr-defined]
                    spec.loader.exec_module(module)  # type: ignore[union-attr]
                    plugin_registry = getattr(module, "registry", None)
                    if isinstance(plugin_registry, ComponentRegistry):
                        before = added
                        for kind in ComponentKind:
                            for alias, info in plugin_registry._store[kind].items():
                                if not registry.has(kind, alias):
                                    registry.register(
                                        kind,
                                        alias,
                                        info.factory,
                                        info.description,
                                        info.requires,
                                    )
                                    added += 1
                        logger.info(
                            "Loaded plugin %s (%d new components)",
                            py_file.name,
                            added - before,
                        )
                except Exception as exc:
                    logger.warning("Failed to load plugin file %s: %s", py_file, exc)

    return added


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

#: The default global :class:`ComponentRegistry`.  Import this instance to
#: register custom components without creating a separate registry.
registry: ComponentRegistry = ComponentRegistry(auto_register_builtins=True)
