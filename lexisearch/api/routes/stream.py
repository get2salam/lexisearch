"""Server-Sent Events (SSE) streaming query route.

Provides ``POST /query/stream`` which returns a ``text/event-stream``
response so the caller receives answer tokens as they are generated.

The route degrades gracefully: if the LLM does not support token-level
streaming, the full answer is sent as a single ``data:`` event followed by
``event: done``.

Event protocol
--------------
Each SSE event has an ``event`` field and a JSON ``data`` payload::

    event: token
    data: {"token": "The ", "index": 0}

    event: token
    data: {"token": "answer ", "index": 1}

    event: source
    data: {"chunk_id": "abc", "doc_id": "xyz", "score": 0.92, "snippet": "…"}

    event: done
    data: {"total_tokens": 12, "latency_ms": 142.3}

    event: error
    data: {"detail": "Pipeline not initialised"}
"""

import json
import time
from typing import Any


def _make_stream_router() -> Any:
    """Build and return the streaming query router (requires FastAPI + sse-starlette)."""
    try:
        from fastapi import APIRouter
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise ImportError("FastAPI is required: pip install 'lexisearch[api]'") from exc

    from lexisearch.api.server import get_pipeline_store

    class StreamQueryBody(BaseModel):
        query: str = Field(..., description="Natural-language question")
        top_k: int = Field(5, ge=1, le=50)
        include_sources: bool = Field(True)
        namespace: str = Field(
            "",
            description="Optional tenant namespace (isolates vector store partition)",
        )

    router = APIRouter(prefix="/query", tags=["streaming"])

    def _sse_event(event: str, data: Any) -> str:
        """Format a single SSE message."""
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {payload}\n\n"

    async def _stream_generator(body: StreamQueryBody) -> Any:
        """Async generator that yields SSE-formatted bytes."""
        store = get_pipeline_store()
        runner = store.get("runner")

        if runner is None:
            yield _sse_event("error", {"detail": "Pipeline not initialised"})
            return

        t0 = time.perf_counter()

        # ---- retrieve context chunks ----------------------------------------
        try:
            retriever = getattr(runner, "retriever", None)
            chunks: list[Any] = (
                retriever.retrieve(body.query, top_k=body.top_k) if retriever is not None else []
            )
        except Exception as exc:
            yield _sse_event("error", {"detail": str(exc)})
            return

        # Emit source events
        if body.include_sources:
            for chunk in chunks:
                yield _sse_event(
                    "source",
                    {
                        "chunk_id": getattr(chunk, "chunk_id", ""),
                        "doc_id": getattr(chunk, "doc_id", ""),
                        "score": float(getattr(chunk, "score", 0.0)),
                        "snippet": str(getattr(chunk, "content", ""))[:300],
                    },
                )

        # ---- generate answer (streaming if supported) -----------------------
        llm = getattr(runner, "llm", None) or getattr(runner, "_llm", None)

        # Try true token-level streaming first
        streamed = False
        total_tokens = 0

        if llm is not None and hasattr(llm, "stream") and callable(getattr(llm, "stream", None)):
            context_text = "\n\n".join(str(getattr(c, "content", "")) for c in chunks)
            prompt = (
                f"Answer the following question using only the provided context.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {body.query}\n\nAnswer:"
            )
            try:
                async for token_text in llm.stream(prompt):
                    if isinstance(token_text, str):
                        yield _sse_event("token", {"token": token_text, "index": total_tokens})
                        total_tokens += 1
                # Only mark as streamed when tokens were actually produced
                if total_tokens > 0:
                    streamed = True
            except Exception:
                streamed = False  # fall through to full-response fallback

        # Fallback: call runner.query() and emit answer as a single burst
        if not streamed:
            try:
                result = runner.query(body.query, top_k=body.top_k)
                answer_text: str = getattr(result, "answer", str(result))
            except Exception as exc:
                yield _sse_event("error", {"detail": str(exc)})
                return

            # Simulate token streaming for clients that expect it
            words = answer_text.split()
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                yield _sse_event("token", {"token": token, "index": i})
                total_tokens += 1

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        yield _sse_event("done", {"total_tokens": total_tokens, "latency_ms": latency_ms})

    @router.post("/stream")
    async def stream_query(body: StreamQueryBody) -> Any:
        """Stream a RAG answer via Server-Sent Events.

        The client should set ``Accept: text/event-stream``.  Each SSE
        event is a JSON object; see module docstring for the full protocol.
        Uses a plain ``StreamingResponse`` with manually-formatted SSE frames
        so behaviour is identical regardless of whether ``sse-starlette``
        is installed.
        """
        from starlette.responses import StreamingResponse

        async def _gen() -> Any:
            async for chunk in _stream_generator(body):
                yield chunk.encode()

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @router.get("/stream/health")
    async def stream_health() -> dict[str, str]:
        """Confirm the streaming endpoint is reachable."""
        try:
            import sse_starlette  # noqa: F401

            sse_status = "available"
        except ImportError:
            sse_status = "fallback (plain StreamingResponse)"

        return {"status": "ok", "sse_backend": sse_status}

    return router
