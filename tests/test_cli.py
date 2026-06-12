"""Tests for the LexiSearch CLI.

Uses Click's CliRunner so no subprocess is spawned.
All tests use mock backends — no external services required.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

click = pytest.importorskip("click", reason="click not installed")

from click.testing import CliRunner  # noqa: E402

from lexisearch.cli.main import cli  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _mock_runner_obj() -> MagicMock:
    """Create a minimal mock PipelineRunner."""
    r = MagicMock()

    query_result = MagicMock()
    query_result.answer = "RAG is Retrieval-Augmented Generation."
    query_result.sources = []
    r.query.return_value = query_result

    r.ingest_documents.return_value = None
    r.retriever = None  # no retriever by default

    pipeline = MagicMock()
    vs = MagicMock()
    vs.__len__ = MagicMock(return_value=0)
    pipeline.vector_store = vs
    r.pipeline = pipeline

    return r


# ---------------------------------------------------------------------------
# ``--help`` and ``--version``
# ---------------------------------------------------------------------------


class TestHelp:
    def test_root_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "index" in result.output
        assert "search" in result.output
        assert "ask" in result.output
        assert "eval" in result.output
        assert "serve" in result.output
        assert "info" in result.output

    def test_index_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["index", "--help"])
        assert result.exit_code == 0
        assert "INDEX" in result.output.upper() or "PATHS" in result.output.upper()

    def test_search_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["search", "--help"])
        assert result.exit_code == 0

    def test_ask_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["ask", "--help"])
        assert result.exit_code == 0

    def test_eval_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0

    def test_serve_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0

    def test_info_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["info", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# ``index`` command
# ---------------------------------------------------------------------------


class TestIndexCommand:
    def test_index_text_file(self, runner: CliRunner, tmp_dir: Path) -> None:
        doc = tmp_dir / "sample.txt"
        doc.write_text("Retrieval-Augmented Generation combines search with LLMs.")

        mock_obj = _mock_runner_obj()
        with patch("lexisearch.cli.main._get_runner", return_value=mock_obj):
            result = runner.invoke(cli, ["index", str(doc)])

        assert result.exit_code == 0
        assert "1 document" in result.output
        mock_obj.ingest_documents.assert_called_once()

    def test_index_multiple_files(self, runner: CliRunner, tmp_dir: Path) -> None:
        for i in range(3):
            (tmp_dir / f"doc{i}.txt").write_text(f"Document number {i} content.")

        mock_obj = _mock_runner_obj()
        with patch("lexisearch.cli.main._get_runner", return_value=mock_obj):
            result = runner.invoke(cli, ["index", str(tmp_dir), "--glob", "*.txt"])

        assert result.exit_code == 0
        assert "3 document" in result.output

    def test_index_directory_no_matches(self, runner: CliRunner, tmp_dir: Path) -> None:
        mock_obj = _mock_runner_obj()
        with patch("lexisearch.cli.main._get_runner", return_value=mock_obj):
            result = runner.invoke(cli, ["index", str(tmp_dir), "--glob", "*.pdf"])

        assert "No documents found" in result.output
        assert "Looked in:" in result.output
        assert "Pattern: '*.pdf'" in result.output
        assert "Try: lexisearch index ./docs" in result.output

    def test_index_nonexistent_path_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["index", "/nonexistent/path/file.txt"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# ``search`` command
# ---------------------------------------------------------------------------


class TestSearchCommand:
    def test_search_no_results(self, runner: CliRunner) -> None:
        mock_obj = _mock_runner_obj()
        with patch("lexisearch.cli.main._get_runner", return_value=mock_obj):
            result = runner.invoke(cli, ["search", "transformer attention"])

        assert result.exit_code == 0
        assert "No results" in result.output
        assert "Next steps:" in result.output
        assert "Index documents first" in result.output
        assert "lexisearch info" in result.output

    def test_search_with_results(self, runner: CliRunner) -> None:
        hit = MagicMock()
        hit.score = 0.85
        hit.content = "Attention mechanisms form the backbone of transformer models."

        mock_obj = _mock_runner_obj()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [hit]
        mock_obj.retriever = mock_retriever

        with patch("lexisearch.cli.main._get_runner", return_value=mock_obj):
            result = runner.invoke(cli, ["search", "attention", "--top-k", "3"])

        assert result.exit_code == 0
        assert "0.8500" in result.output

    def test_search_json_output(self, runner: CliRunner) -> None:
        mock_obj = _mock_runner_obj()
        with patch("lexisearch.cli.main._get_runner", return_value=mock_obj):
            result = runner.invoke(cli, ["search", "test query", "--json"])

        assert result.exit_code == 0
        # Should be valid JSON
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# ``ask`` command
# ---------------------------------------------------------------------------


class TestAskCommand:
    def test_ask_basic(self, runner: CliRunner) -> None:
        mock_obj = _mock_runner_obj()
        with patch("lexisearch.cli.main._get_runner", return_value=mock_obj):
            result = runner.invoke(cli, ["ask", "What is RAG?"])

        assert result.exit_code == 0
        assert "RAG is Retrieval-Augmented Generation" in result.output

    def test_ask_json_output(self, runner: CliRunner) -> None:
        mock_obj = _mock_runner_obj()
        with patch("lexisearch.cli.main._get_runner", return_value=mock_obj):
            result = runner.invoke(cli, ["ask", "What is RAG?", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "answer" in parsed
        assert "question" in parsed
        assert "latency_ms" in parsed

    def test_ask_includes_latency(self, runner: CliRunner) -> None:
        mock_obj = _mock_runner_obj()
        with patch("lexisearch.cli.main._get_runner", return_value=mock_obj):
            result = runner.invoke(cli, ["ask", "test question"])

        assert result.exit_code == 0
        assert "Latency" in result.output or "ms" in result.output

    def test_ask_with_sources(self, runner: CliRunner) -> None:
        source = MagicMock()
        source.score = 0.92
        source.content = "RAG combines dense retrieval with autoregressive generation."

        mock_obj = _mock_runner_obj()
        result_obj = MagicMock()
        result_obj.answer = "RAG is Retrieval-Augmented Generation."
        result_obj.sources = [source]
        mock_obj.query.return_value = result_obj

        with patch("lexisearch.cli.main._get_runner", return_value=mock_obj):
            result = runner.invoke(cli, ["ask", "Explain RAG"])

        assert result.exit_code == 0
        assert "Sources" in result.output

    def test_ask_calls_runner_query(self, runner: CliRunner) -> None:
        mock_obj = _mock_runner_obj()
        with patch("lexisearch.cli.main._get_runner", return_value=mock_obj):
            runner.invoke(cli, ["ask", "my question", "--top-k", "7"])

        mock_obj.query.assert_called_once_with("my question", top_k=7)


# ---------------------------------------------------------------------------
# ``eval`` command
# ---------------------------------------------------------------------------


class TestEvalCommand:
    def _write_jsonl(self, path: Path, samples: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(json.dumps(s) + "\n")

    def test_eval_basic(self, runner: CliRunner, tmp_dir: Path) -> None:
        dataset = tmp_dir / "samples.jsonl"
        self._write_jsonl(
            dataset,
            [
                {
                    "question": "What is RAG?",
                    "contexts": ["RAG stands for Retrieval-Augmented Generation."],
                    "answer": "RAG is Retrieval-Augmented Generation.",
                    "reference": "RAG stands for Retrieval-Augmented Generation.",
                }
            ],
        )
        result = runner.invoke(cli, ["eval", str(dataset)])
        assert result.exit_code == 0
        assert "Evaluation Results" in result.output

    def test_eval_specific_metrics(self, runner: CliRunner, tmp_dir: Path) -> None:
        dataset = tmp_dir / "samples.jsonl"
        self._write_jsonl(
            dataset,
            [
                {
                    "question": "What is FAISS?",
                    "contexts": ["FAISS enables similarity search."],
                    "answer": "FAISS enables similarity search.",
                    "reference": "FAISS enables similarity search.",
                }
            ],
        )
        result = runner.invoke(cli, ["eval", str(dataset), "--metrics", "exact_match,token_f1"])
        assert result.exit_code == 0
        assert "exact_match" in result.output

    def test_eval_json_output(self, runner: CliRunner, tmp_dir: Path) -> None:
        dataset = tmp_dir / "samples.jsonl"
        self._write_jsonl(
            dataset,
            [
                {
                    "question": "Test?",
                    "contexts": ["Test context."],
                    "answer": "Test answer.",
                    "reference": "Test reference.",
                }
            ],
        )
        result = runner.invoke(cli, ["eval", str(dataset), "--metrics", "token_f1", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "aggregate" in parsed

    def test_eval_unknown_metric_exits(self, runner: CliRunner, tmp_dir: Path) -> None:
        dataset = tmp_dir / "samples.jsonl"
        self._write_jsonl(
            dataset,
            [{"question": "q", "contexts": ["c"], "answer": "a"}],
        )
        result = runner.invoke(cli, ["eval", str(dataset), "--metrics", "bogus_metric"])
        assert result.exit_code != 0

    def test_eval_skips_invalid_lines(self, runner: CliRunner, tmp_dir: Path) -> None:
        dataset = tmp_dir / "samples.jsonl"
        with dataset.open("w") as fh:
            fh.write("not valid json\n")
            fh.write(
                json.dumps(
                    {
                        "question": "q",
                        "contexts": ["c"],
                        "answer": "a",
                        "reference": "r",
                    }
                )
                + "\n"
            )

        result = runner.invoke(cli, ["eval", str(dataset), "--metrics", "token_f1"])
        # Should succeed with 1 valid sample
        assert result.exit_code == 0

    def test_eval_empty_file_exits(self, runner: CliRunner, tmp_dir: Path) -> None:
        dataset = tmp_dir / "empty.jsonl"
        dataset.write_text("")
        result = runner.invoke(cli, ["eval", str(dataset)])
        assert result.exit_code != 0

    def test_eval_multiple_samples(self, runner: CliRunner, tmp_dir: Path) -> None:
        dataset = tmp_dir / "multi.jsonl"
        samples = [
            {
                "question": f"Question {i}",
                "contexts": [f"Context {i}."],
                "answer": f"Answer {i}.",
                "reference": f"Reference {i}.",
            }
            for i in range(5)
        ]
        self._write_jsonl(dataset, samples)
        result = runner.invoke(cli, ["eval", str(dataset), "--metrics", "token_f1"])
        assert result.exit_code == 0
        assert "5" in result.output  # "5 samples" in the report heading


# ---------------------------------------------------------------------------
# ``info`` command
# ---------------------------------------------------------------------------


class TestInfoCommand:
    def test_info_runs(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0

    def test_info_shows_version(self, runner: CliRunner) -> None:
        from lexisearch import __version__

        result = runner.invoke(cli, ["info"])
        assert __version__ in result.output

    def test_info_shows_embedders(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["info"])
        assert "Embedders" in result.output or "embedder" in result.output.lower()

    def test_info_shows_mock_backend(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["info"])
        assert "mock" in result.output.lower()


# ---------------------------------------------------------------------------
# ``_get_runner`` caching
# ---------------------------------------------------------------------------


class TestGetRunner:
    def test_get_runner_returns_same_object(self) -> None:
        import lexisearch.cli.main as cli_module

        # Reset the singleton
        cli_module._runner = None

        r1 = cli_module._get_runner()
        r2 = cli_module._get_runner()
        assert r1 is r2

        # Cleanup
        cli_module._runner = None

    def test_get_runner_creates_pipeline(self) -> None:
        import lexisearch.cli.main as cli_module

        cli_module._runner = None
        r = cli_module._get_runner(embedder="mock", llm="mock")
        assert r is not None
        cli_module._runner = None
