"""Tests for the LexiSearch FastAPI API layer.

Uses the FastAPI TestClient (requires httpx and fastapi to be installed).
All tests use mock backends — no external services required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest

# ---------------------------------------------------------------------------
# Skip all tests if FastAPI or httpx is not installed
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
httpx = pytest.importorskip("httpx", reason="httpx not installed")


from fastapi.testclient import TestClient  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_runner() -> MagicMock:
    """Return a mock PipelineRunner."""
    runner = MagicMock()

    # query returns a simple result object
    query_result = MagicMock()
    query_result.answer = "RAG combines retrieval with generation."
    query_result.sources = []
    runner.query.return_value = query_result

    # ingest does nothing
    runner.ingest_documents.return_value = None

    # pipeline has a vector_store
    vs = MagicMock()
    vs.__len__ = MagicMock(return_value=5)
    vs.embedding_dim = 384
    runner.pipeline.vector_store = vs

    return runner


@pytest.fixture()
def app(mock_runner: MagicMock) -> Any:
    """Create a fresh FastAPI app with pipeline store pre-populated."""
    from lexisearch.api.server import _pipeline_store, create_app

    # Patch the builder so lifespan doesn't call _build_runner
    with patch("lexisearch.api.server._build_runner", return_value=mock_runner):
        application = create_app()

    # Manually populate store (bypass lifespan for tests)
    _pipeline_store.clear()
    _pipeline_store["runner"] = mock_runner
    _pipeline_store["vector_store_type"] = "memory"
    _pipeline_store["embedder_type"] = "mock"
    _pipeline_store["llm_type"] = "mock"

    return application


@pytest.fixture()
def client(app: Any, mock_runner: MagicMock) -> Generator[TestClient, None, None]:
    """Return a synchronous TestClient (lifespan patched to use mock runner)."""
    with (
        patch("lexisearch.api.server._build_runner", return_value=mock_runner),
        TestClient(app, raise_server_exceptions=True) as c,
    ):
        yield c


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body_has_status(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "healthy"

    def test_health_body_has_version(self, client: TestClient) -> None:
        from lexisearch import __version__

        body = client.get("/health").json()
        assert body["version"] == __version__

    def test_health_body_has_components(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert "components" in body
        assert "embedder" in body["components"]

    def test_stats_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health/stats")
        assert resp.status_code == 200

    def test_stats_body_fields(self, client: TestClient) -> None:
        body = client.get("/health/stats").json()
        assert "total_chunks" in body
        assert "vector_store" in body

    def test_root_redirect(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "docs" in body


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------


class TestDocumentEndpoints:
    def test_ingest_basic(self, client: TestClient, mock_runner: MagicMock) -> None:
        payload = {
            "content": "RAG stands for Retrieval-Augmented Generation.",
            "title": "RAG Overview",
        }
        resp = client.post("/documents", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "doc_id" in body
        mock_runner.ingest_documents.assert_called_once()

    def test_ingest_with_metadata(self, client: TestClient) -> None:
        payload = {
            "content": "Vector databases store embeddings efficiently.",
            "title": "Vector DBs",
            "source": "https://example.com/vectordb",
            "metadata": {"category": "infrastructure", "year": 2024},
        }
        resp = client.post("/documents", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ingest_missing_content_returns_422(self, client: TestClient) -> None:
        resp = client.post("/documents", json={"title": "No content"})
        assert resp.status_code == 422

    def test_ingest_empty_content(self, client: TestClient) -> None:
        # Empty string is valid — pipeline handles it
        payload = {"content": ""}
        resp = client.post("/documents", json=payload)
        assert resp.status_code == 200

    def test_delete_document(self, client: TestClient, mock_runner: MagicMock) -> None:
        doc_id = "test-doc-001"
        resp = client.delete(f"/documents/{doc_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_id"] == doc_id
        assert body["status"] == "ok"

    def test_ingest_no_runner_returns_503(self, app: Any, mock_runner: MagicMock) -> None:
        from lexisearch.api.server import _pipeline_store

        # Override runner with None AFTER lifespan sets it
        def _null_runner() -> None:
            return None  # type: ignore[return-value]

        with (
            patch("lexisearch.api.server._build_runner", side_effect=_null_runner),
            TestClient(app) as c,
        ):
            _pipeline_store["runner"] = None  # Force None after startup
            resp = c.post("/documents", json={"content": "hello"})
            assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Query endpoints
# ---------------------------------------------------------------------------


class TestQueryEndpoints:
    def test_query_basic(self, client: TestClient, mock_runner: MagicMock) -> None:
        payload = {"query": "What is RAG?"}
        resp = client.post("/query", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "answer" in body
        assert body["query"] == "What is RAG?"
        mock_runner.query.assert_called_once()

    def test_query_latency_present(self, client: TestClient) -> None:
        resp = client.post("/query", json={"query": "test query"})
        assert resp.status_code == 200
        assert resp.json()["latency_ms"] >= 0

    def test_query_timing_header(self, client: TestClient) -> None:
        resp = client.post("/query", json={"query": "test"})
        assert "x-process-time-ms" in resp.headers

    def test_query_top_k_respected(self, client: TestClient, mock_runner: MagicMock) -> None:
        client.post("/query", json={"query": "test", "top_k": 10})
        call_kwargs = mock_runner.query.call_args
        assert call_kwargs is not None

    def test_query_no_runner_returns_503(self, app: Any, mock_runner: MagicMock) -> None:
        from lexisearch.api.server import _pipeline_store

        with (
            patch("lexisearch.api.server._build_runner", return_value=mock_runner),
            TestClient(app) as c,
        ):
            _pipeline_store["runner"] = None  # Force None after startup
            resp = c.post("/query", json={"query": "test"})
            assert resp.status_code == 503

    def test_search_endpoint(self, client: TestClient) -> None:
        resp = client.get("/query/search", params={"q": "embeddings", "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert "hits" in body
        assert body["query"] == "embeddings"

    def test_query_missing_query_returns_422(self, client: TestClient) -> None:
        resp = client.post("/query", json={"top_k": 5})
        assert resp.status_code == 422

    def test_query_empty_string_returns_422(self, client: TestClient) -> None:
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422

    def test_query_whitespace_only_returns_422(self, client: TestClient) -> None:
        resp = client.post("/query", json={"query": "   \n\t "})
        assert resp.status_code == 422
        assert "empty or whitespace" in resp.text

    def test_query_strips_surrounding_whitespace(
        self, client: TestClient, mock_runner: MagicMock
    ) -> None:
        resp = client.post("/query", json={"query": "  What is RAG?  "})
        assert resp.status_code == 200
        mock_runner.query.assert_called_once_with("What is RAG?", top_k=5)

    def test_search_blank_q_returns_422(self, client: TestClient) -> None:
        resp = client.get("/query/search", params={"q": "   "})
        assert resp.status_code == 422
        assert "empty or whitespace" in resp.text


# ---------------------------------------------------------------------------
# Evaluate endpoints
# ---------------------------------------------------------------------------


class TestEvaluateEndpoints:
    def _sample_payload(self) -> dict[str, Any]:
        return {
            "samples": [
                {
                    "question": "What is RAG?",
                    "contexts": ["RAG stands for Retrieval-Augmented Generation."],
                    "answer": "RAG is Retrieval-Augmented Generation.",
                    "reference": "RAG stands for Retrieval-Augmented Generation.",
                }
            ],
            "metrics": ["exact_match", "token_f1"],
        }

    def test_evaluate_basic(self, client: TestClient) -> None:
        resp = client.post("/evaluate", json=self._sample_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["num_samples"] == 1
        assert "aggregate" in body

    def test_evaluate_aggregate_has_requested_metrics(self, client: TestClient) -> None:
        resp = client.post("/evaluate", json=self._sample_payload())
        agg = resp.json()["aggregate"]
        assert "exact_match" in agg
        assert "token_f1" in agg

    def test_evaluate_scores_in_range(self, client: TestClient) -> None:
        resp = client.post("/evaluate", json=self._sample_payload())
        agg = resp.json()["aggregate"]
        for score in agg.values():
            assert 0.0 <= score <= 1.0

    def test_evaluate_all_metrics_default(self, client: TestClient) -> None:
        payload = {
            "samples": [
                {
                    "question": "What is FAISS?",
                    "contexts": ["FAISS is a library for efficient similarity search."],
                    "answer": "FAISS enables similarity search.",
                    "reference": "FAISS is a library for efficient similarity search.",
                }
            ]
        }
        resp = client.post("/evaluate", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["aggregate"]) >= 1

    def test_evaluate_unknown_metric_returns_422(self, client: TestClient) -> None:
        payload = {**self._sample_payload(), "metrics": ["nonexistent_metric"]}
        resp = client.post("/evaluate", json=payload)
        assert resp.status_code == 422

    def test_evaluate_empty_samples_returns_422(self, client: TestClient) -> None:
        resp = client.post("/evaluate", json={"samples": []})
        assert resp.status_code == 422

    def test_list_metrics(self, client: TestClient) -> None:
        resp = client.get("/evaluate/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "metrics" in body
        assert "faithfulness" in body["metrics"]
        assert "token_f1" in body["metrics"]

    def test_evaluate_multiple_samples(self, client: TestClient) -> None:
        samples = [
            {
                "question": f"Question {i}",
                "contexts": [f"Context about topic {i}."],
                "answer": f"Answer {i}.",
                "reference": f"Reference {i}.",
            }
            for i in range(5)
        ]
        resp = client.post("/evaluate", json={"samples": samples, "metrics": ["token_f1"]})
        assert resp.status_code == 200
        assert resp.json()["num_samples"] == 5


# ---------------------------------------------------------------------------
# Schemas module (no FastAPI dependency)
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_ingest_request_defaults(self) -> None:
        from lexisearch.api.schemas import IngestRequest

        req = IngestRequest(content="hello")
        assert req.chunk_size == 512
        assert req.chunk_overlap == 64
        assert req.metadata == {}

    def test_query_request_defaults(self) -> None:
        from lexisearch.api.schemas import QueryRequest

        req = QueryRequest(query="test")
        assert req.top_k == 5
        assert req.include_sources is True
        assert req.stream is False

    def test_eval_request_defaults(self) -> None:
        from lexisearch.api.schemas import EvalRequest, EvalSampleRequest

        req = EvalRequest(
            samples=[
                EvalSampleRequest(
                    question="q",
                    contexts=["c"],
                    answer="a",
                )
            ]
        )
        assert req.metrics == []

    def test_health_response_defaults(self) -> None:
        from lexisearch.api.schemas import HealthResponse

        hr = HealthResponse()
        assert hr.status == "healthy"
        assert hr.components == {}
