"""Compare BM25 vs Hybrid retrieval quality with offline IR metrics.

Runs a fully offline evaluation of two retrieval strategies on a small
annotated corpus.  Uses the deterministic MockEmbedder so no API keys
or model downloads are required.

Outputs an NDCG@5 / MRR / MAP / Recall@5 / P@5 comparison table.

Run::

    python examples/offline_retrieval_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexisearch.chunking import RecursiveChunker
from lexisearch.embeddings import MockEmbedder
from lexisearch.evaluation.ir_metrics import compute_ir_metrics
from lexisearch.models import Document, DocumentMetadata
from lexisearch.retrieval import (
    BM25Retriever,
    FusionMethod,
    HybridConfig,
    HybridRetriever,
    VectorRetriever,
)
from lexisearch.vectorstore import InMemoryVectorStore, VectorStoreConfig

DIMENSIONS = 64
TOP_K = 5

# ---------------------------------------------------------------------------
# Corpus — eight short documents covering three topic clusters
# ---------------------------------------------------------------------------

CORPUS: list[tuple[str, str, str]] = [
    # (doc_id, title, content)
    (
        "doc-chunking",
        "Text chunking strategies",
        "Fixed-size, recursive, sentence, and semantic chunking strategies "
        "determine how documents are split before embedding.",
    ),
    (
        "doc-embedding",
        "Embedding models",
        "Sentence transformers produce dense vector embeddings that capture "
        "semantic similarity between texts for retrieval tasks.",
    ),
    (
        "doc-bm25",
        "BM25 sparse retrieval",
        "BM25 ranks documents using term frequency and inverse document "
        "frequency with a length normalisation penalty for sparse keyword search.",
    ),
    (
        "doc-hybrid",
        "Hybrid search and RRF",
        "Hybrid retrieval fuses BM25 and vector scores via RRF or linear "
        "combination to capture both lexical and semantic relevance.",
    ),
    (
        "doc-reranking",
        "Cross-encoder reranking",
        "A cross-encoder reranker re-scores a candidate set produced by a "
        "first-stage retriever to improve precision at small rank cutoffs.",
    ),
    (
        "doc-evaluation",
        "Retrieval evaluation metrics",
        "NDCG, MRR, MAP, and Recall@k are standard metrics for measuring "
        "the quality of ranked retrieval results against relevance judgements.",
    ),
    (
        "doc-latency",
        "Latency budgets and SLOs",
        "Production RAG services should define p95 latency budgets and "
        "circuit-breaker thresholds to maintain acceptable response times.",
    ),
    (
        "doc-faithfulness",
        "Answer faithfulness",
        "Faithfulness measures whether every claim in the generated answer "
        "is supported by the retrieved context passages.",
    ),
]

# ---------------------------------------------------------------------------
# Ground-truth relevance judgements: query → set of relevant doc-ids
# ---------------------------------------------------------------------------

QUERIES: list[tuple[str, set[str]]] = [
    (
        "how are documents split for embedding",
        {"doc-chunking", "doc-embedding"},
    ),
    (
        "keyword-based sparse ranking with term frequency",
        {"doc-bm25", "doc-hybrid"},
    ),
    (
        "measure retrieval quality NDCG and MRR",
        {"doc-evaluation"},
    ),
    (
        "response latency SLO production deployment",
        {"doc-latency"},
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_retrievers(
    chunks: list,
    embedded_chunks: list,
    embedder: MockEmbedder,
) -> tuple[BM25Retriever, HybridRetriever]:
    bm25 = BM25Retriever()
    bm25.add_chunks(chunks)

    config = VectorStoreConfig(collection_name="eval-demo", dimensions=DIMENSIONS)
    store = InMemoryVectorStore(config=config)
    store.initialize()
    store.add(embedded_chunks)
    vector = VectorRetriever(store, embedder)

    hybrid = HybridRetriever(
        retrievers=[bm25, vector],
        config=HybridConfig(
            fusion_method=FusionMethod.RRF,
            weights=[0.6, 0.4],
            top_k=TOP_K,
        ),
    )
    return bm25, hybrid


def _retrieve_doc_ids(retriever: BM25Retriever | HybridRetriever, query: str) -> list[str]:
    """Return ordered list of document IDs for a retriever's top-k results."""
    results = retriever.retrieve(query, top_k=TOP_K)
    seen: dict[str, None] = {}
    for r in results:
        doc_id = r.chunk.document_id
        seen.setdefault(doc_id, None)
    return list(seen.keys())[:TOP_K]


def _row(label: str, report) -> str:
    return (
        f"  {label:<10}  "
        f"NDCG@{TOP_K}={report.ndcg_at_k:.3f}  "
        f"MRR={report.mrr:.3f}  "
        f"MAP={report.map_score:.3f}  "
        f"R@{TOP_K}={report.recall_at_k:.3f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build corpus, run retrieval, print IR metric comparison."""
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=0)
    embedder = MockEmbedder(dimensions=DIMENSIONS)

    # Build documents with stable IDs
    documents = [
        Document(
            content=content,
            metadata=DocumentMetadata(source=f"{doc_id}.md", title=title),
            id=doc_id,
        )
        for doc_id, title, content in CORPUS
    ]

    chunks = [chunk for doc in documents for chunk in chunker.chunk(doc)]
    embedded_chunks = embedder.embed_chunks(chunks)

    bm25, hybrid = _build_retrievers(chunks, embedded_chunks, embedder)

    all_relevant = [rel for _, rel in QUERIES]
    bm25_retrieved: list[list[str]] = []
    hybrid_retrieved: list[list[str]] = []

    for query, _ in QUERIES:
        bm25_retrieved.append(_retrieve_doc_ids(bm25, query))
        hybrid_retrieved.append(_retrieve_doc_ids(hybrid, query))

    bm25_report = compute_ir_metrics(bm25_retrieved, all_relevant, k=TOP_K)
    hybrid_report = compute_ir_metrics(hybrid_retrieved, all_relevant, k=TOP_K)

    print(f"Corpus: {len(documents)} docs  Queries: {len(QUERIES)}  k={TOP_K}")
    print()
    print("Retrieval quality comparison")
    print("=" * 62)
    print(_row("BM25", bm25_report))
    print(_row("Hybrid", hybrid_report))
    print("=" * 62)

    ndcg_delta = hybrid_report.ndcg_at_k - bm25_report.ndcg_at_k
    verdict = "Hybrid wins" if ndcg_delta >= 0 else "BM25 wins"
    print(f"NDCG@{TOP_K} delta (Hybrid - BM25): {ndcg_delta:+.3f}  →  {verdict}")


if __name__ == "__main__":
    main()
