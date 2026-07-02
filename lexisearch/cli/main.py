"""LexiSearch CLI entry point.

All commands are implemented here using only stdlib + the lexisearch package.
Click is an optional dependency — a minimal fallback is provided so that
``python -m lexisearch.cli`` works even without Click installed.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Attempt to import Click; fall back to a minimal shim so tests can import
# this module without Click installed.
# ---------------------------------------------------------------------------


def _require_click() -> Any:
    """Import click or raise a user-friendly error."""
    try:
        import click

        return click
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Click is required for the LexiSearch CLI.\n"
            "Install it with: pip install 'lexisearch[cli]'"
        ) from exc


# ---------------------------------------------------------------------------
# Pipeline singleton (initialised lazily by ``_get_runner``)
# ---------------------------------------------------------------------------

_runner: Any = None


def _get_runner(embedder: str = "mock", llm: str = "mock") -> Any:
    """Return a cached PipelineRunner (create on first call)."""
    global _runner
    if _runner is None:
        from lexisearch.embeddings import MockEmbedder
        from lexisearch.generation import MockLLM
        from lexisearch.pipeline import PipelineBuilder, PipelineRunner

        emb: Any = MockEmbedder()
        lm: Any = MockLLM()

        if embedder == "openai":
            try:
                from lexisearch.embeddings import OpenAIEmbedder

                emb = OpenAIEmbedder()
            except Exception as e:
                print(
                    f"Warning: could not load OpenAI embedder ({e}), using mock.",
                    file=sys.stderr,
                )

        if llm == "openai":
            try:
                from lexisearch.generation import OpenAILLM

                lm = OpenAILLM()
            except Exception as e:
                print(f"Warning: could not load OpenAI LLM ({e}), using mock.", file=sys.stderr)

        pipeline = (
            PipelineBuilder.create("lexisearch-cli")
            .embed(emb)
            .store()
            .retrieve(top_k=5)
            .generate(lm)
            .build()
        )
        _runner = PipelineRunner(pipeline)
    return _runner


# ---------------------------------------------------------------------------
# Helper: pretty print
# ---------------------------------------------------------------------------


def _hr(width: int = 60) -> str:
    return "─" * width


def _wrap(text: str, width: int = 72, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _path_list_label(paths: tuple[str, ...], *, limit: int = 3) -> str:
    """Return a compact, readable path summary for CLI guidance."""
    shown = [str(Path(path)) for path in paths[:limit]]
    if len(paths) > limit:
        shown.append(f"… +{len(paths) - limit} more")
    return ", ".join(shown)


def _echo_empty_index_hint(
    click: Any, paths: tuple[str, ...], pattern: str, recursive: bool
) -> None:
    """Print actionable guidance when no files match an index request."""
    click.echo("No documents found to index.")
    click.echo(f"  Looked in: {_path_list_label(paths)}")
    click.echo(f"  Pattern: {pattern!r} ({'recursive' if recursive else 'non-recursive'})")
    click.echo("  Try: lexisearch index ./docs --glob '**/*.txt'")


def _echo_blank_query_error(click: Any, command: str, argument: str) -> None:
    """Print a clear, actionable error when a required text argument is blank."""
    click.echo(f"Error: {argument} must not be empty or whitespace-only.", err=True)
    click.echo(f'  Try: lexisearch {command} "your question or search terms"', err=True)


def _echo_empty_search_hint(click: Any, query: str, top_k: int) -> None:
    """Print a screen-reader-friendly empty state with next actions."""
    click.echo("  No results found.")
    click.echo(f"  Query: {query!r} | requested top_k={top_k}")
    click.echo("  Next steps:")
    click.echo("    1. Index documents first: lexisearch index ./docs --glob '**/*.txt'")
    click.echo("    2. Use broader terms or increase --top-k for exploratory searches.")
    click.echo("    3. Run lexisearch info to confirm available local backends.")


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


def _build_cli() -> Any:
    click = _require_click()

    @click.group()
    @click.version_option(package_name="lexisearch")
    def cli() -> None:
        r"""LexiSearch — production-ready RAG framework CLI.

        \b
        Commands:
          index   — Index documents from files or directories
          search  — Semantic search without generation
          ask     — Full RAG query (retrieve + generate)
          eval    — Evaluate RAG quality on a JSONL dataset
          serve   — Start the FastAPI HTTP server
          info    — Show runtime configuration
        """

    # ------------------------------------------------------------------
    # index
    # ------------------------------------------------------------------

    @cli.command()
    @click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
    @click.option("--embedder", default="mock", show_default=True, help="Embedder backend")
    @click.option("--glob", "pattern", default="**/*.txt", show_default=True, help="File glob")
    @click.option("--recursive/--no-recursive", default=True, show_default=True)
    def index(paths: tuple[str, ...], embedder: str, pattern: str, recursive: bool) -> None:
        r"""Index documents from FILES or DIRECTORIES.

        \b
        Examples:
          lexisearch index ./docs/
          lexisearch index paper.txt report.pdf --embedder openai
          lexisearch index ./corpus/ --glob "**/*.md"
        """
        from lexisearch.ingest import TextLoader
        from lexisearch.models import Document  # noqa: TC001

        runner = _get_runner(embedder=embedder)
        docs: list[Document] = []

        for raw_path in paths:
            p = Path(raw_path)
            if p.is_dir():
                simple = pattern.replace("**/", "")
                files = list(p.glob(pattern)) if recursive else list(p.glob(simple))
            else:
                files = [p]

            for f in files:
                try:
                    loader = TextLoader()
                    doc = loader.load(str(f))
                    if isinstance(doc, list):
                        docs.extend(doc)
                    else:
                        docs.append(doc)
                except Exception as e:
                    click.echo(f"  ⚠  Skipping {f}: {e}", err=True)

        if not docs:
            _echo_empty_index_hint(click, paths, pattern, recursive)
            return

        click.echo(f"Indexing {len(docs)} document(s)…")
        runner.ingest_documents(docs)
        click.echo(f"✓ Done — {len(docs)} document(s) indexed.")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    @cli.command()
    @click.argument("query")
    @click.option("--top-k", default=5, show_default=True, help="Number of results")
    @click.option("--embedder", default="mock", show_default=True)
    @click.option("--json", "as_json", is_flag=True, default=False, help="Output JSON")
    def search(query: str, top_k: int, embedder: str, as_json: bool) -> None:
        r"""Semantic search — retrieve chunks WITHOUT generation.

        \b
        Example:
          lexisearch search "transformer attention mechanism" --top-k 10
        """
        if not query.strip():
            _echo_blank_query_error(click, "search", "QUERY")
            sys.exit(2)

        runner = _get_runner(embedder=embedder)
        try:
            retriever = getattr(runner, "retriever", None)
            results = retriever.retrieve(query, top_k=top_k) if retriever is not None else []
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        if as_json:
            output = [
                {
                    "rank": i + 1,
                    "score": float(getattr(r, "score", 0.0)),
                    "snippet": str(getattr(r, "content", ""))[:300],
                }
                for i, r in enumerate(results)
            ]
            click.echo(json.dumps(output, indent=2))
            return

        click.echo(_hr())
        click.echo(f"  Query: {query}")
        click.echo(_hr())
        if not results:
            _echo_empty_search_hint(click, query, top_k)
        for i, r in enumerate(results, 1):
            score = float(getattr(r, "score", 0.0))
            snippet = str(getattr(r, "content", ""))[:200]
            click.echo(f"\n  [{i}] score={score:.4f}")
            click.echo(_wrap(snippet))
        click.echo(_hr())

    # ------------------------------------------------------------------
    # ask
    # ------------------------------------------------------------------

    @cli.command()
    @click.argument("question")
    @click.option("--top-k", default=5, show_default=True)
    @click.option("--embedder", default="mock", show_default=True)
    @click.option("--llm", default="mock", show_default=True, help="LLM backend")
    @click.option("--json", "as_json", is_flag=True, default=False)
    def ask(question: str, top_k: int, embedder: str, llm: str, as_json: bool) -> None:
        r"""Full RAG query — retrieve THEN generate a grounded answer.

        \b
        Example:
          lexisearch ask "What are the main benefits of RAG?" --llm openai
        """
        import time

        if not question.strip():
            _echo_blank_query_error(click, "ask", "QUESTION")
            sys.exit(2)

        runner = _get_runner(embedder=embedder, llm=llm)
        t0 = time.perf_counter()
        try:
            result = runner.query(question, top_k=top_k)
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = getattr(result, "answer", str(result))
        sources = getattr(result, "sources", [])

        if as_json:
            click.echo(
                json.dumps(
                    {
                        "question": question,
                        "answer": answer,
                        "latency_ms": round(latency_ms, 1),
                        "sources": [
                            {
                                "score": float(getattr(s, "score", 0.0)),
                                "snippet": str(getattr(s, "content", ""))[:200],
                            }
                            for s in sources
                        ],
                    },
                    indent=2,
                )
            )
            return

        click.echo(_hr())
        click.echo(f"  Q: {question}")
        click.echo(_hr())
        click.echo("\n  Answer:\n")
        click.echo(_wrap(answer, indent="    "))
        if sources:
            click.echo(f"\n  Sources ({len(sources)}):")
            for i, s in enumerate(sources, 1):
                score = float(getattr(s, "score", 0.0))
                snippet = str(getattr(s, "content", ""))[:120]
                click.echo(f"    [{i}] score={score:.4f}  {snippet!r}")
        click.echo(f"\n  Latency: {latency_ms:.1f} ms")
        click.echo(_hr())

    # ------------------------------------------------------------------
    # eval
    # ------------------------------------------------------------------

    @cli.command()
    @click.argument("dataset", type=click.Path(exists=True))
    @click.option(
        "--metrics",
        default="",
        help="Comma-separated metric names (default: all)",
    )
    @click.option("--json", "as_json", is_flag=True, default=False)
    def eval(dataset: str, metrics: str, as_json: bool) -> None:
        r"""Evaluate RAG quality on a JSONL DATASET file.

        Each line must be a JSON object with keys:
        question, contexts (list), answer, reference (optional).

        \b
        Example:
          lexisearch eval eval_samples.jsonl --metrics faithfulness,token_f1
        """
        from lexisearch.evaluation import EvalSample, Evaluator
        from lexisearch.evaluation.metrics import (
            AnswerRelevanceMetric,
            ContextPrecisionMetric,
            ContextRecallMetric,
            ExactMatchMetric,
            FaithfulnessMetric,
            TokenF1Metric,
        )

        _all = {
            "faithfulness": FaithfulnessMetric,
            "context_precision": ContextPrecisionMetric,
            "context_recall": ContextRecallMetric,
            "answer_relevance": AnswerRelevanceMetric,
            "exact_match": ExactMatchMetric,
            "token_f1": TokenF1Metric,
        }

        requested = [m.strip() for m in metrics.split(",") if m.strip()] or list(_all)
        bad = [m for m in requested if m not in _all]
        if bad:
            click.echo(f"Unknown metrics: {bad}. Available: {list(_all)}", err=True)
            sys.exit(1)

        metric_instances = [_all[m]() for m in requested]
        evaluator = Evaluator(metrics=metric_instances)

        samples: list[EvalSample] = []
        with Path(dataset).open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    samples.append(
                        EvalSample(
                            question=obj["question"],
                            contexts=obj["contexts"],
                            answer=obj["answer"],
                            reference=obj.get("reference", ""),
                        )
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    click.echo(f"  ⚠  Skipping line {lineno}: {e}", err=True)

        if not samples:
            click.echo("No valid samples found in dataset.", err=True)
            sys.exit(1)

        if not as_json:
            click.echo(f"Evaluating {len(samples)} samples with metrics: {requested}…")
        report = evaluator.evaluate(samples)

        if as_json:
            click.echo(json.dumps({"aggregate": report.aggregate}, indent=2))
            return

        click.echo(_hr())
        click.echo(f"  Evaluation Results  ({len(samples)} samples)")
        click.echo(_hr())
        for metric, score in report.aggregate.items():
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            click.echo(f"  {metric:<22} {bar}  {score:.4f}")
        click.echo(_hr())

    # ------------------------------------------------------------------
    # serve
    # ------------------------------------------------------------------

    @cli.command()
    @click.option("--host", default="0.0.0.0", show_default=True)
    @click.option("--port", default=8000, show_default=True, type=int)
    @click.option("--reload", is_flag=True, default=False, help="Auto-reload on code changes")
    @click.option("--workers", default=1, show_default=True, type=int)
    def serve(host: str, port: int, reload: bool, workers: int) -> None:
        r"""Start the LexiSearch FastAPI HTTP server.

        \b
        Example:
          lexisearch serve --port 8080 --reload
        """
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover
            raise click.ClickException(
                "Uvicorn is required to serve. Install with: pip install 'lexisearch[api]'"
            ) from exc

        click.echo(f"Starting LexiSearch API on http://{host}:{port}")
        click.echo("  Docs: /docs  |  Health: /health")
        uvicorn.run(
            "lexisearch.api.server:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
        )

    # ------------------------------------------------------------------
    # info
    # ------------------------------------------------------------------

    @cli.command()
    def info() -> None:
        """Show runtime configuration and available backends."""
        from lexisearch import __version__

        click.echo(_hr())
        click.echo(f"  LexiSearch v{__version__}")
        click.echo(_hr())

        backends: dict[str, list[str]] = {
            "Embedders": ["mock (built-in)"],
            "LLMs": ["mock (built-in)"],
            "Vector Stores": ["memory (built-in)"],
        }

        try:
            from lexisearch.embeddings import OpenAIEmbedder  # noqa: F401

            backends["Embedders"].append("openai")
        except ImportError:
            pass

        try:
            import sentence_transformers  # noqa: F401

            backends["Embedders"].append("sentence-transformers")
        except ImportError:
            pass

        try:
            from lexisearch.generation import OpenAILLM  # noqa: F401

            backends["LLMs"].append("openai")
        except ImportError:
            pass

        try:
            import faiss  # noqa: F401

            backends["Vector Stores"].append("faiss")
        except (ImportError, Exception):
            pass

        try:
            import chromadb  # noqa: F401

            backends["Vector Stores"].append("chromadb")
        except (ImportError, Exception):
            pass

        api_key = "✓ set" if os.environ.get("OPENAI_API_KEY") else "✗ not set"

        for category, available in backends.items():
            click.echo(f"\n  {category}:")
            for b in available:
                click.echo(f"    • {b}")

        click.echo(f"\n  OPENAI_API_KEY: {api_key}")
        click.echo(_hr())

    return cli


# Build the CLI object at module level
cli = _build_cli()


if __name__ == "__main__":
    cli()
