"""Latency benchmarks for core LexiSearch pipeline components.

Measures wall-clock latency (P50, P95, P99, mean) for:
- MockEmbedder.embed_text()         — baseline embedding cost
- InMemoryVectorStore.search()      — brute-force search at various scales
- Fixed-size chunker                — document splitting throughput
- Full pipeline query (mock)        — end-to-end RAG latency

Usage::

    python benchmarks/bench_latency.py
    python benchmarks/bench_latency.py --runs 200 --verbose

Requirements: only the lexisearch package itself (no external dependencies).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _timed(fn, *args, runs: int = 100, **kwargs):
    """Run *fn* *runs* times and return latency statistics (seconds)."""
    latencies: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        latencies.append(time.perf_counter() - t0)
    latencies.sort()
    return {
        "runs": runs,
        "mean_ms": statistics.mean(latencies) * 1000,
        "median_ms": statistics.median(latencies) * 1000,
        "p95_ms": latencies[int(0.95 * runs)] * 1000,
        "p99_ms": latencies[int(0.99 * runs)] * 1000,
        "min_ms": latencies[0] * 1000,
        "max_ms": latencies[-1] * 1000,
    }


def _print_stats(label: str, stats: dict) -> None:
    print(
        f"  {label:<40} "
        f"mean={stats['mean_ms']:6.3f}ms  "
        f"p50={stats['median_ms']:6.3f}ms  "
        f"p95={stats['p95_ms']:6.3f}ms  "
        f"p99={stats['p99_ms']:6.3f}ms"
    )


# ---------------------------------------------------------------------------
# Benchmark: embedding
# ---------------------------------------------------------------------------


def bench_embedding(runs: int = 200) -> None:
    from lexisearch.embeddings import MockEmbedder

    embedder = MockEmbedder()
    texts = [
        "This is a short sentence.",
        "Retrieval Augmented Generation (RAG) combines retrieval with generation.",
        "The transformer architecture introduced self-attention mechanisms.",
        "Large language models are trained on diverse internet-scale corpora.",
    ]

    print("\n[Embedding latency]")
    for text in texts:
        stats = _timed(embedder.embed_text, text, runs=runs)
        _print_stats(repr(text[:40]), stats)


# ---------------------------------------------------------------------------
# Benchmark: vector search at various index sizes
# ---------------------------------------------------------------------------


def bench_vector_search(runs: int = 100) -> None:
    from lexisearch.models import Chunk, ChunkStrategy, EmbeddedChunk, Embedding
    from lexisearch.vectorstore import InMemoryVectorStore, VectorStoreConfig

    dim = 128
    config = VectorStoreConfig(dimensions=dim)

    def _make_item(i: int) -> EmbeddedChunk:
        import random

        chunk = Chunk(
            content=f"Chunk content number {i}",
            document_id=f"doc-{i // 10}",
            index=i % 10,
            start_char=i * 100,
            end_char=(i + 1) * 100,
            metadata={"idx": i},
            strategy=ChunkStrategy.FIXED_SIZE,
            id=f"chunk-{i}",
        )
        vector = [random.gauss(0, 1) for _ in range(dim)]
        return EmbeddedChunk(
            chunk=chunk,
            embedding=Embedding(chunk_id=chunk.id, vector=vector, model="test"),
        )

    import random

    random.seed(42)
    query_vec = [random.gauss(0, 1) for _ in range(dim)]

    print("\n[Vector search latency vs index size]")
    for n_items in (100, 500, 1_000, 5_000, 10_000):
        store = InMemoryVectorStore(config=config)
        store.initialize()
        items = [_make_item(i) for i in range(n_items)]
        store.add(items)

        stats = _timed(store.search, query_vec, 10, runs=runs)
        _print_stats(f"n={n_items:>6} items, top_k=10", stats)


# ---------------------------------------------------------------------------
# Benchmark: chunking throughput
# ---------------------------------------------------------------------------


def bench_chunking(runs: int = 50) -> None:
    from lexisearch.chunking import FixedSizeChunker

    # Synthetic documents of varying lengths
    from lexisearch.models import Document, DocumentMetadata

    short_doc = Document(
        content=" ".join(["word"] * 200),
        id="short",
        metadata=DocumentMetadata(title="Short document"),
    )
    medium_doc = Document(
        content=" ".join(["word"] * 2_000),
        id="medium",
        metadata=DocumentMetadata(title="Medium document"),
    )
    long_doc = Document(
        content=" ".join(["word"] * 10_000),
        id="long",
        metadata=DocumentMetadata(title="Long document"),
    )

    chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=64)

    print("\n[Chunking latency]")
    for doc in (short_doc, medium_doc, long_doc):
        n_words = len(doc.content.split())
        stats = _timed(chunker.chunk, doc, runs=runs)
        _print_stats(f"{doc.title} ({n_words} words)", stats)


# ---------------------------------------------------------------------------
# Benchmark: full pipeline (mock, end-to-end)
# ---------------------------------------------------------------------------


def bench_pipeline(runs: int = 100) -> None:
    from lexisearch.embeddings import MockEmbedder
    from lexisearch.generation import MockLLM
    from lexisearch.models import Document
    from lexisearch.pipeline import PipelineBuilder, PipelineRunner

    emb = MockEmbedder()
    llm = MockLLM()

    pipeline = (
        PipelineBuilder.create("bench-pipeline")
        .embed(emb)
        .store()
        .retrieve(top_k=5)
        .generate(llm)
        .build()
    )
    runner = PipelineRunner(pipeline)

    # Pre-load some documents
    from lexisearch.models import DocumentMetadata

    docs = [
        Document(
            content=f"This document covers topic {i}. " * 20,
            id=f"doc-{i}",
            metadata=DocumentMetadata(title=f"Document {i}"),
        )
        for i in range(50)
    ]
    runner.ingest_documents(docs)

    queries = [
        "What are the main topics covered?",
        "Explain the key concepts in detail.",
        "Summarise the most important findings.",
    ]

    print("\n[Full RAG pipeline latency (mock LLM)]")
    for q in queries:
        stats = _timed(runner.query, q, top_k=5, runs=runs)
        _print_stats(repr(q[:50]), stats)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LexiSearch latency benchmark suite")
    parser.add_argument("--runs", type=int, default=100, help="Iterations per benchmark")
    parser.add_argument("--verbose", action="store_true", help="Extra output")
    parser.add_argument(
        "--suite",
        choices=["embedding", "search", "chunking", "pipeline", "all"],
        default="all",
        help="Which benchmark suite to run",
    )
    args = parser.parse_args()

    sep = "=" * 72
    print(sep)
    print("  LexiSearch Latency Benchmarks")
    print(f"  runs={args.runs}")
    print(sep)

    if args.suite in ("embedding", "all"):
        bench_embedding(runs=args.runs)

    if args.suite in ("chunking", "all"):
        bench_chunking(runs=min(args.runs, 50))

    if args.suite in ("search", "all"):
        bench_vector_search(runs=args.runs)

    if args.suite in ("pipeline", "all"):
        bench_pipeline(runs=args.runs)

    print(f"\n{sep}")
    print("  Done.")
    print(sep)


if __name__ == "__main__":
    main()
