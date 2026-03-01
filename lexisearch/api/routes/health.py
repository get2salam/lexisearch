"""Health-check and index-statistics routes."""

from __future__ import annotations

from typing import Any


def _make_health_router() -> Any:
    """Build and return the health router (requires FastAPI)."""
    try:
        from fastapi import APIRouter
    except ImportError as exc:  # pragma: no cover
        raise ImportError("FastAPI is required for the API server: pip install fastapi") from exc

    from lexisearch import __version__
    from lexisearch.api.schemas import HealthResponse, IndexStatsResponse
    from lexisearch.api.server import get_pipeline_store

    router = APIRouter(prefix="/health", tags=["health"])

    @router.get("", response_model=None)
    async def health() -> dict[str, Any]:  # type: ignore[misc]
        """Liveness probe — always returns 200 when the server is running."""
        store = get_pipeline_store()
        components: dict[str, str] = {
            "vector_store": store.get("vector_store_type", "memory"),
            "embedder": store.get("embedder_type", "mock"),
            "llm": store.get("llm_type", "mock"),
        }
        resp = HealthResponse(
            status="healthy",
            version=__version__,
            components=components,
        )
        return resp.__dict__

    @router.get("/stats", response_model=None)
    async def index_stats() -> dict[str, Any]:  # type: ignore[misc]
        """Return statistics about the in-memory vector index."""
        store = get_pipeline_store()
        runner = store.get("runner")
        total_docs = 0
        total_chunks = 0
        dim = 0
        if runner is not None:
            try:
                vs = runner.pipeline.vector_store  # type: ignore[attr-defined]
                total_chunks = len(vs)
                dim = vs.embedding_dim if hasattr(vs, "embedding_dim") else 0
            except Exception:
                pass
        resp = IndexStatsResponse(
            total_documents=total_docs,
            total_chunks=total_chunks,
            embedding_dim=dim,
            vector_store=store.get("vector_store_type", "memory"),
            embedder=store.get("embedder_type", "mock"),
        )
        return resp.__dict__

    return router


# Lazy export — callers do ``from lexisearch.api.routes.health import router``
def _get_router() -> Any:  # pragma: no cover
    return _make_health_router()
