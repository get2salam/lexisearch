"""LexiSearch REST API sub-package.

Provides a FastAPI application that exposes the full LexiSearch RAG pipeline
over HTTP.

Quick start::

    # Install optional deps
    # pip install 'lexisearch[api]'

    # Run with uvicorn
    # uvicorn lexisearch.api.server:app --reload --port 8000

    # Or programmatically:
    from lexisearch.api import create_app
    app = create_app()

Endpoints
---------
GET  /health          — liveness probe
GET  /health/stats    — index statistics
POST /documents       — ingest a document
DELETE /documents/{id} — remove a document
POST /query           — full RAG query (retrieve + generate)
GET  /query/search    — semantic search only (no generation)
POST /evaluate        — batch RAG evaluation
GET  /evaluate/metrics — list available metric names
"""

from __future__ import annotations

from lexisearch.api.server import create_app, get_pipeline_store

__all__ = ["create_app", "get_pipeline_store"]
