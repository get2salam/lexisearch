"""Tests for the SSE streaming query endpoint (/query/stream).

The tests exercise:
- POST /query/stream returns text/event-stream content
- SSE events are well-formed JSON (token / source / done / error)
- Stream health endpoint
- Graceful degradation when pipeline is uninitialised
- Token ordering and count in done event
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse(raw: str) -> list[dict[str, Any]]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("event:"):
            current["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            payload = line[len("data:") :].strip()
            try:
                current["data"] = json.loads(payload)
            except json.JSONDecodeError:
                current["data"] = payload
        elif line == "" and current:
            events.append(current)
            current = {}

    if current:
        events.append(current)

    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_runner() -> MagicMock:
    """Minimal PipelineRunner mock with a working query() method."""
    runner = MagicMock()

    # query() result
    result = MagicMock()
    result.answer = "The answer is forty-two."
    result.sources = []
    runner.query.return_value = result

    # retriever
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    runner.retriever = retriever

    return runner


@pytest.fixture()
def app_client(mock_runner: MagicMock) -> Any:
    """Starlette TestClient with the LexiSearch FastAPI app."""
    pytest.importorskip("fastapi")
    pytest.importorskip("starlette")

    from starlette.testclient import TestClient

    from lexisearch.api.server import _pipeline_store, create_app

    _pipeline_store.clear()
    _pipeline_store["runner"] = mock_runner
    _pipeline_store["embedder_type"] = "mock"
    _pipeline_store["llm_type"] = "mock"

    test_app = create_app()
    # Re-inject runner after app creation (lifespan won't run in TestClient by default)
    _pipeline_store["runner"] = mock_runner

    return TestClient(test_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Stream health endpoint
# ---------------------------------------------------------------------------


class TestStreamHealth:
    def test_health_ok(self, app_client: Any) -> None:
        resp = app_client.get("/query/stream/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "sse_backend" in body

    def test_health_sse_backend_field(self, app_client: Any) -> None:
        resp = app_client.get("/query/stream/health")
        assert "sse_backend" in resp.json()


# ---------------------------------------------------------------------------
# POST /query/stream — basic SSE structure
# ---------------------------------------------------------------------------


class TestStreamEndpointBasic:
    def test_returns_event_stream_content_type(self, app_client: Any) -> None:
        resp = app_client.post(
            "/query/stream",
            json={"query": "What is the meaning of life?"},
        )
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/event-stream" in ct

    def test_done_event_present(self, app_client: Any) -> None:
        resp = app_client.post(
            "/query/stream",
            json={"query": "Explain transformers."},
        )
        events = _parse_sse(resp.text)
        event_types = [e.get("event") for e in events]
        assert "done" in event_types

    def test_token_events_present(self, app_client: Any) -> None:
        resp = app_client.post(
            "/query/stream",
            json={"query": "Summarise the document."},
        )
        events = _parse_sse(resp.text)
        token_events = [e for e in events if e.get("event") == "token"]
        assert len(token_events) > 0

    def test_token_index_is_sequential(self, app_client: Any) -> None:
        resp = app_client.post(
            "/query/stream",
            json={"query": "What are the key findings?"},
        )
        events = _parse_sse(resp.text)
        token_events = [e for e in events if e.get("event") == "token"]
        indices = [e["data"]["index"] for e in token_events]
        assert indices == list(range(len(indices)))

    def test_done_total_tokens_matches_token_events(self, app_client: Any) -> None:
        resp = app_client.post(
            "/query/stream",
            json={"query": "Give me a summary."},
        )
        events = _parse_sse(resp.text)
        token_count = sum(1 for e in events if e.get("event") == "token")
        done_events = [e for e in events if e.get("event") == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["total_tokens"] == token_count

    def test_done_latency_ms_positive(self, app_client: Any) -> None:
        resp = app_client.post(
            "/query/stream",
            json={"query": "What is RAG?"},
        )
        events = _parse_sse(resp.text)
        done = next(e for e in events if e.get("event") == "done")
        assert done["data"]["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# Source events
# ---------------------------------------------------------------------------


class TestStreamSourceEvents:
    def test_no_source_events_when_retriever_returns_empty(self, app_client: Any) -> None:
        resp = app_client.post(
            "/query/stream",
            json={"query": "Test", "include_sources": True},
        )
        events = _parse_sse(resp.text)
        source_events = [e for e in events if e.get("event") == "source"]
        assert len(source_events) == 0

    def test_source_events_when_chunks_available(self, mock_runner: MagicMock) -> None:
        pytest.importorskip("fastapi")

        # Inject a retriever that returns one fake chunk
        chunk = MagicMock()
        chunk.chunk_id = "c1"
        chunk.doc_id = "d1"
        chunk.score = 0.9
        chunk.content = "Relevant context here."
        mock_runner.retriever.retrieve.return_value = [chunk]

        from starlette.testclient import TestClient

        from lexisearch.api.server import _pipeline_store, create_app

        _pipeline_store.clear()
        _pipeline_store["runner"] = mock_runner
        _pipeline_store["embedder_type"] = "mock"
        _pipeline_store["llm_type"] = "mock"
        app = create_app()
        _pipeline_store["runner"] = mock_runner

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/query/stream",
            json={"query": "What does the document say?", "include_sources": True},
        )
        events = _parse_sse(resp.text)
        source_events = [e for e in events if e.get("event") == "source"]
        assert len(source_events) == 1
        assert source_events[0]["data"]["chunk_id"] == "c1"
        assert source_events[0]["data"]["score"] == pytest.approx(0.9)

    def test_no_source_events_when_include_sources_false(self, mock_runner: MagicMock) -> None:
        pytest.importorskip("fastapi")

        chunk = MagicMock()
        chunk.chunk_id = "c2"
        chunk.doc_id = "d2"
        chunk.score = 0.75
        chunk.content = "Some content."
        mock_runner.retriever.retrieve.return_value = [chunk]

        from starlette.testclient import TestClient

        from lexisearch.api.server import _pipeline_store, create_app

        _pipeline_store.clear()
        _pipeline_store["runner"] = mock_runner
        _pipeline_store["embedder_type"] = "mock"
        _pipeline_store["llm_type"] = "mock"
        app = create_app()
        _pipeline_store["runner"] = mock_runner

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/query/stream",
            json={"query": "Any question", "include_sources": False},
        )
        events = _parse_sse(resp.text)
        source_events = [e for e in events if e.get("event") == "source"]
        assert len(source_events) == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestStreamErrorHandling:
    def test_error_event_when_pipeline_absent(self) -> None:
        pytest.importorskip("fastapi")

        from starlette.testclient import TestClient

        from lexisearch.api.server import _pipeline_store, create_app

        _pipeline_store.clear()
        # No runner injected
        app = create_app()
        # Don't inject runner this time

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/query/stream", json={"query": "Test"})
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) >= 1
        assert "detail" in error_events[0]["data"]

    def test_error_event_when_query_raises(self, mock_runner: MagicMock) -> None:
        pytest.importorskip("fastapi")

        mock_runner.retriever.retrieve.side_effect = RuntimeError("index corrupt")

        from starlette.testclient import TestClient

        from lexisearch.api.server import _pipeline_store, create_app

        _pipeline_store.clear()
        _pipeline_store["runner"] = mock_runner
        _pipeline_store["embedder_type"] = "mock"
        _pipeline_store["llm_type"] = "mock"
        app = create_app()
        _pipeline_store["runner"] = mock_runner

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/query/stream", json={"query": "Will this fail?"})
        events = _parse_sse(resp.text)
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) >= 1


# ---------------------------------------------------------------------------
# Namespace field
# ---------------------------------------------------------------------------


class TestStreamNamespace:
    def test_namespace_field_accepted(self, app_client: Any) -> None:
        resp = app_client.post(
            "/query/stream",
            json={"query": "Namespace test", "namespace": "tenant-abc"},
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        assert any(e.get("event") == "done" for e in events)

    def test_empty_namespace_field_accepted(self, app_client: Any) -> None:
        resp = app_client.post(
            "/query/stream",
            json={"query": "Empty namespace", "namespace": ""},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# _parse_sse helper unit tests
# ---------------------------------------------------------------------------


class TestParseSseHelper:
    def test_parse_single_event(self) -> None:
        raw = 'event: done\ndata: {"total_tokens": 5}\n\n'
        events = _parse_sse(raw)
        assert len(events) == 1
        assert events[0]["event"] == "done"
        assert events[0]["data"]["total_tokens"] == 5

    def test_parse_multiple_events(self) -> None:
        raw = (
            'event: token\ndata: {"token": "Hello ", "index": 0}\n\n'
            'event: token\ndata: {"token": "world", "index": 1}\n\n'
            'event: done\ndata: {"total_tokens": 2, "latency_ms": 10.0}\n\n'
        )
        events = _parse_sse(raw)
        assert len(events) == 3
        assert events[0]["event"] == "token"
        assert events[2]["event"] == "done"

    def test_parse_empty_string(self) -> None:
        assert _parse_sse("") == []

    def test_parse_invalid_json_data(self) -> None:
        raw = "event: raw\ndata: not-json\n\n"
        events = _parse_sse(raw)
        assert events[0]["data"] == "not-json"
