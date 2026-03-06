"""Throughput benchmarks for the LexiSearch ingestion and retrieval pipeline.

Measures documents/second and queries/second at various concurrency levels.

Metrics reported:
- Ingest throughput: documents indexed per second
- Retrieval throughput: queries answered per second (single thread)
- Concurrent retrieval throughput: queries/sec with N threads

Usage::

    python benchmarks/bench_throughput.py
    python benchmarks/bench_throughput.py --docs 500 --queries 200

Requirements: stdlib + lexisearch package only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _banner(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _throughput(n_ops: int, elapsed_s: float) -> float:
    return n_ops / elapsed_s if elapsed_s > 0 else float("inf")


# ---------------------------------------------------------------------------
# Benchmark: ingest throughput
# ---------------------------------------------------------------------------


def bench_ingest_throughput(n_docs: int = 200) -> None:
    from lexisearch.embeddings import MockEmbedder
    from lexisearch.generation import MockLLM
    from lexisearch.models import Document
    from lexisearch.pipeline import PipelineBuilder, PipelineRunner

    emb = MockEmbedder()
    llm = MockLLM()
    pipeline = (
        PipelineBuilder.create("throughput-ingest")
        .embed(emb)
        .store()
        .retrieve(top_k=5)
        .generate(llm)
        .build()
    )
    runner = PipelineRunner(pipeline)

    from lexisearch.models import DocumentMetadata

    docs = [
        Document(
            content=(
                f"Document {i}: This research paper explores the application of "
                f"neural retrieval methods in domain-specific knowledge bases. "
                f"Section {i % 5 + 1} discusses the experimental results and ablation studies. "
            ),
            id=f"bench-doc-{i}",
            metadata=DocumentMetadata(title=f"Research Paper {i}"),
        )
        for i in range(n_docs)
    ]

    _banner("Ingest Throughput")

    # Batch ingest (all at once)
    t0 = time.perf_counter()
    runner.ingest_documents(docs)
    elapsed = time.perf_counter() - t0
    tps = _throughput(n_docs, elapsed)
    print(f"  Batch ingest:  {n_docs} docs in {elapsed * 1000:.1f}ms  →  {tps:.1f} docs/sec")

    # Per-document ingest (incremental)
    pipeline2 = (
        PipelineBuilder.create("throughput-ingest-2")
        .embed(emb)
        .store()
        .retrieve(top_k=5)
        .generate(llm)
        .build()
    )
    runner2 = PipelineRunner(pipeline2)
    t0 = time.perf_counter()
    for doc in docs:
        runner2.ingest_documents([doc])
    elapsed = time.perf_counter() - t0
    tps = _throughput(n_docs, elapsed)
    print(f"  Per-doc ingest:{n_docs} docs in {elapsed * 1000:.1f}ms  →  {tps:.1f} docs/sec")


# ---------------------------------------------------------------------------
# Benchmark: retrieval throughput (single thread)
# ---------------------------------------------------------------------------


def bench_retrieval_throughput(n_docs: int = 200, n_queries: int = 100) -> None:
    from lexisearch.embeddings import MockEmbedder
    from lexisearch.generation import MockLLM
    from lexisearch.models import Document
    from lexisearch.pipeline import PipelineBuilder, PipelineRunner

    emb = MockEmbedder()
    llm = MockLLM()
    pipeline = (
        PipelineBuilder.create("throughput-retrieval")
        .embed(emb)
        .store()
        .retrieve(top_k=5)
        .generate(llm)
        .build()
    )
    runner = PipelineRunner(pipeline)

    from lexisearch.models import DocumentMetadata

    docs = [
        Document(
            content=f"Research findings on topic {i}: transformer models achieve SOTA results.",
            id=f"doc-{i}",
            metadata=DocumentMetadata(title=f"Paper {i}"),
        )
        for i in range(n_docs)
    ]
    runner.ingest_documents(docs)

    queries = [
        "What are the main findings of the research?",
        "How do transformer models perform on benchmark tasks?",
        "What methodology was used in the experiments?",
        "Compare retrieval-augmented and fine-tuning approaches.",
        "Summarise the conclusions and future work.",
    ]

    _banner("Retrieval Throughput (single thread)")

    for top_k in (1, 5, 10):
        t0 = time.perf_counter()
        for i in range(n_queries):
            runner.query(queries[i % len(queries)], top_k=top_k)
        elapsed = time.perf_counter() - t0
        qps = _throughput(n_queries, elapsed)
        print(
            f"  top_k={top_k:<3}  {n_queries} queries in {elapsed * 1000:.1f}ms"
            f"  →  {qps:.1f} queries/sec"
        )


# ---------------------------------------------------------------------------
# Benchmark: concurrent retrieval throughput
# ---------------------------------------------------------------------------


def bench_concurrent_retrieval(n_docs: int = 200, n_queries: int = 100) -> None:
    from lexisearch.embeddings import MockEmbedder
    from lexisearch.generation import MockLLM
    from lexisearch.models import Document
    from lexisearch.pipeline import PipelineBuilder, PipelineRunner

    emb = MockEmbedder()
    llm = MockLLM()
    pipeline = (
        PipelineBuilder.create("throughput-concurrent")
        .embed(emb)
        .store()
        .retrieve(top_k=5)
        .generate(llm)
        .build()
    )
    runner = PipelineRunner(pipeline)

    from lexisearch.models import DocumentMetadata

    docs = [
        Document(
            content=f"Document {i} with important research context and findings.",
            id=f"doc-{i}",
            metadata=DocumentMetadata(title=f"Doc {i}"),
        )
        for i in range(n_docs)
    ]
    runner.ingest_documents(docs)

    queries = [f"Query about topic {i}" for i in range(n_queries)]

    _banner("Concurrent Retrieval Throughput")

    for n_workers in (1, 2, 4):

        def _run_query(q: str) -> None:
            runner.query(q, top_k=5)

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_run_query, queries))
        elapsed = time.perf_counter() - t0
        qps = _throughput(n_queries, elapsed)
        print(
            f"  workers={n_workers}  {n_queries} queries in {elapsed * 1000:.1f}ms"
            f"  →  {qps:.1f} queries/sec"
        )


# ---------------------------------------------------------------------------
# Benchmark: embedding batch throughput
# ---------------------------------------------------------------------------


def bench_embedding_throughput(n_texts: int = 500) -> None:
    from lexisearch.embeddings import MockEmbedder

    embedder = MockEmbedder()
    texts = [f"Sample text for embedding number {i}." for i in range(n_texts)]

    _banner("Embedding Throughput")

    # Single-text calls
    t0 = time.perf_counter()
    for text in texts:
        embedder.embed_text(text)
    elapsed = time.perf_counter() - t0
    tps = _throughput(n_texts, elapsed)
    print(f"  Single-text:  {n_texts} texts in {elapsed * 1000:.1f}ms  →  {tps:.1f} texts/sec")

    # Batch call (if supported)
    if hasattr(embedder, "embed_batch"):
        t0 = time.perf_counter()
        embedder.embed_batch(texts)
        elapsed = time.perf_counter() - t0
        tps = _throughput(n_texts, elapsed)
        print(f"  Batch embed:  {n_texts} texts in {elapsed * 1000:.1f}ms  →  {tps:.1f} texts/sec")
    else:
        print("  Batch embed:  not supported by this embedder")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LexiSearch throughput benchmark")
    parser.add_argument("--docs", type=int, default=200)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--texts", type=int, default=500)
    parser.add_argument(
        "--suite",
        choices=["ingest", "retrieval", "concurrent", "embedding", "all"],
        default="all",
    )
    args = parser.parse_args()

    sep = "=" * 60
    print(sep)
    print("  LexiSearch Throughput Benchmarks")
    print(sep)

    if args.suite in ("ingest", "all"):
        bench_ingest_throughput(n_docs=args.docs)

    if args.suite in ("embedding", "all"):
        bench_embedding_throughput(n_texts=args.texts)

    if args.suite in ("retrieval", "all"):
        bench_retrieval_throughput(n_docs=args.docs, n_queries=args.queries)

    if args.suite in ("concurrent", "all"):
        bench_concurrent_retrieval(n_docs=args.docs, n_queries=args.queries)

    print(f"\n{sep}")
    print("  Done.")
    print(sep)


if __name__ == "__main__":
    main()
