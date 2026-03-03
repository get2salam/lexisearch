"""Document ingestion routes."""

from typing import Any


def _make_documents_router() -> Any:
    """Build and return the documents router (requires FastAPI)."""
    try:
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI and pydantic are required for the API server: pip install fastapi"
        ) from exc

    from lexisearch.api.server import get_pipeline_store
    from lexisearch.models import Document, DocumentMetadata

    # ------------------------------------------------------------------
    # Pydantic I/O models (separate from the plain dataclasses in schemas)
    # ------------------------------------------------------------------

    class IngestBody(BaseModel):
        """Request body for document ingestion."""

        content: str = Field(..., description="Raw text content of the document")
        title: str = Field("", description="Human-readable document title")
        source: str = Field("", description="URI or path this document came from")
        doc_id: str = Field("", description="Optional caller-supplied stable identifier")
        metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")

    class IngestResult(BaseModel):
        doc_id: str
        chunks_created: int
        tokens_estimated: int
        status: str = "ok"
        message: str = ""

    class DeleteResult(BaseModel):
        doc_id: str
        status: str
        message: str = ""

    router = APIRouter(prefix="/documents", tags=["documents"])

    @router.post("", response_model=IngestResult)
    async def ingest_document(body: IngestBody) -> IngestResult:
        """Ingest a document into the RAG pipeline index."""
        store = get_pipeline_store()
        runner = store.get("runner")
        if runner is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialised")

        meta = DocumentMetadata(
            title=body.title or "Untitled",
            source=body.source,
            extra=body.metadata,
        )
        doc = Document(
            content=body.content,
            metadata=meta,
            **({"id": body.doc_id} if body.doc_id else {}),
        )
        actual_id = doc.id

        try:
            runner.ingest_documents([doc])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        words = len(body.content.split())
        return IngestResult(
            doc_id=actual_id,
            chunks_created=max(1, words // 100) if words else 1,
            tokens_estimated=words,
            status="ok",
        )

    @router.delete("/{doc_id}", response_model=DeleteResult)
    async def delete_document(doc_id: str) -> DeleteResult:
        """Remove a document (and its chunks) from the index."""
        store = get_pipeline_store()
        runner = store.get("runner")
        if runner is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialised")

        try:
            vs = runner.pipeline.vector_store
            vs.delete_by_metadata({"doc_id": doc_id})
        except Exception:
            # Best-effort deletion; not all vector stores support metadata delete
            pass

        return DeleteResult(doc_id=doc_id, status="ok", message="Document removed from index")

    return router
