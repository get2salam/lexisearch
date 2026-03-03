"""Query / search routes."""

import time
from typing import Any


def _make_query_router() -> Any:
    """Build and return the query router (requires FastAPI)."""
    try:
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI and pydantic are required for the API server: pip install fastapi"
        ) from exc

    from lexisearch.api.server import get_pipeline_store

    class QueryBody(BaseModel):
        query: str = Field(..., description="Natural-language question or keyword query")
        top_k: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve")
        filters: dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
        include_sources: bool = Field(True, description="Include source attribution")

    class SourceItem(BaseModel):
        chunk_id: str
        doc_id: str
        title: str
        score: float
        snippet: str

    class QueryResult(BaseModel):
        answer: str
        sources: list[SourceItem] = []
        query: str = ""
        latency_ms: float = 0.0
        status: str = "ok"
        message: str = ""

    router = APIRouter(prefix="/query", tags=["query"])

    @router.post("", response_model=QueryResult)
    async def query(body: QueryBody) -> QueryResult:
        """Run a full RAG pipeline query and return a grounded answer."""
        store = get_pipeline_store()
        runner = store.get("runner")
        if runner is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialised")

        t0 = time.perf_counter()
        try:
            result = runner.query(body.query, top_k=body.top_k)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        latency = (time.perf_counter() - t0) * 1000

        sources: list[SourceItem] = []
        if body.include_sources and hasattr(result, "sources"):
            for src in result.sources:
                sources.append(
                    SourceItem(
                        chunk_id=getattr(src, "chunk_id", ""),
                        doc_id=getattr(src, "doc_id", ""),
                        title=getattr(src, "title", ""),
                        score=float(getattr(src, "score", 0.0)),
                        snippet=getattr(src, "snippet", "")[:300],
                    )
                )

        return QueryResult(
            answer=getattr(result, "answer", str(result)),
            sources=sources,
            query=body.query,
            latency_ms=round(latency, 2),
            status="ok",
        )

    @router.get("/search", response_model=None)
    async def search(q: str, top_k: int = 5) -> dict[str, Any]:
        """Lightweight keyword/semantic search (no generation step)."""
        store = get_pipeline_store()
        runner = store.get("runner")
        if runner is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialised")

        t0 = time.perf_counter()
        try:
            # Use retriever directly if available
            retriever = getattr(runner, "retriever", None)
            results = retriever.retrieve(q, top_k=top_k) if retriever is not None else []
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        latency = (time.perf_counter() - t0) * 1000
        hits = [
            {
                "chunk_id": getattr(r, "chunk_id", ""),
                "score": float(getattr(r, "score", 0.0)),
                "snippet": str(getattr(r, "content", ""))[:300],
            }
            for r in results
        ]
        return {"query": q, "hits": hits, "latency_ms": round(latency, 2)}

    return router
