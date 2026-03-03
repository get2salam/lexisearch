"""LexiSearch FastAPI application factory.

Usage (programmatic)::

    from lexisearch.api.server import create_app
    app = create_app()

Usage (uvicorn CLI)::

    uvicorn lexisearch.api.server:app --reload

Environment variables
---------------------
LEXISEARCH_EMBEDDER
    ``"mock"`` (default) or ``"openai"`` or ``"sentence-transformers"``.
LEXISEARCH_LLM
    ``"mock"`` (default) or ``"openai"``.
OPENAI_API_KEY
    Required when either embedder or LLM is ``"openai"``.
LEXISEARCH_LOG_LEVEL
    Logging level string, default ``"INFO"``.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger("lexisearch.api")

# ---------------------------------------------------------------------------
# Shared mutable state (singleton per process)
# ---------------------------------------------------------------------------

_pipeline_store: dict[str, Any] = {}


def get_pipeline_store() -> dict[str, Any]:
    """Return the global pipeline store (thread-safe for reads)."""
    return _pipeline_store


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def _build_runner() -> Any:
    """Construct a PipelineRunner from environment configuration."""
    from lexisearch.embeddings import MockEmbedder
    from lexisearch.generation import MockLLM
    from lexisearch.pipeline import PipelineBuilder, PipelineRunner

    embedder_type = os.environ.get("LEXISEARCH_EMBEDDER", "mock").lower()
    llm_type = os.environ.get("LEXISEARCH_LLM", "mock").lower()

    # Choose embedder
    if embedder_type == "openai":
        try:
            from lexisearch.embeddings import OpenAIEmbedder

            embedder: Any = OpenAIEmbedder()
        except Exception as exc:
            logger.warning("Could not initialise OpenAI embedder (%s), falling back to mock", exc)
            embedder = MockEmbedder()
            embedder_type = "mock"
    else:
        embedder = MockEmbedder()
        embedder_type = "mock"

    # Choose LLM
    if llm_type == "openai":
        try:
            from lexisearch.generation import OpenAILLM

            llm: Any = OpenAILLM()
        except Exception as exc:
            logger.warning("Could not initialise OpenAI LLM (%s), falling back to mock", exc)
            llm = MockLLM()
            llm_type = "mock"
    else:
        llm = MockLLM()
        llm_type = "mock"

    pipeline = (
        PipelineBuilder.create("lexisearch-api")
        .embed(embedder)
        .store()
        .retrieve(top_k=5)
        .generate(llm)
        .build()
    )

    _pipeline_store["vector_store_type"] = "memory"
    _pipeline_store["embedder_type"] = embedder_type
    _pipeline_store["llm_type"] = llm_type

    return PipelineRunner(pipeline)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(*, title: str = "LexiSearch API", version: str | None = None) -> Any:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    title:
        OpenAPI title shown in the interactive docs.
    version:
        API version string; defaults to the package version.

    Returns:
        fastapi.FastAPI
        Configured application instance.
    """
    try:
        from fastapi import FastAPI, Request, Response
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI is required for the API server.\n"
            "Install it with: pip install 'lexisearch[api]'"
        ) from exc

    from lexisearch import __version__

    api_version = version or __version__

    @asynccontextmanager
    async def _lifespan(application: Any) -> AsyncGenerator[None, None]:
        """Startup / shutdown lifecycle."""
        log_level = os.environ.get("LEXISEARCH_LOG_LEVEL", "INFO").upper()
        logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
        logger.info("LexiSearch API v%s starting up…", api_version)
        runner = _build_runner()
        _pipeline_store["runner"] = runner
        logger.info(
            "Pipeline ready — embedder=%s llm=%s",
            _pipeline_store["embedder_type"],
            _pipeline_store["llm_type"],
        )
        yield
        logger.info("LexiSearch API shutting down")
        _pipeline_store.clear()

    app = FastAPI(
        title=title,
        version=api_version,
        description=(
            "LexiSearch — a production-ready RAG framework API.\n\n"
            "Endpoints for document ingestion, semantic search, grounded Q&A, "
            "and evaluation."
        ),
        lifespan=_lifespan,
    )

    # CORS — allow all origins by default (configure via env in production)
    origins = os.environ.get("LEXISEARCH_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request-timing middleware
    @app.middleware("http")
    async def _add_timing(request: Request, call_next: Any) -> Response:
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # Mount routers
    from lexisearch.api.routes.documents import _make_documents_router
    from lexisearch.api.routes.evaluate import _make_evaluate_router
    from lexisearch.api.routes.health import _make_health_router
    from lexisearch.api.routes.query import _make_query_router

    app.include_router(_make_health_router())
    app.include_router(_make_documents_router())
    app.include_router(_make_query_router())
    app.include_router(_make_evaluate_router())

    @app.get("/", include_in_schema=False)
    async def _root() -> JSONResponse:
        return JSONResponse(
            {
                "name": "LexiSearch API",
                "version": api_version,
                "docs": "/docs",
                "health": "/health",
            }
        )

    return app


# ---------------------------------------------------------------------------
# Module-level ``app`` for ``uvicorn lexisearch.api.server:app``
# ---------------------------------------------------------------------------

try:
    app = create_app()
except ImportError:  # pragma: no cover
    app = None
