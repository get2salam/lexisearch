"""Run a fully offline hybrid search demo.

This example uses the deterministic MockEmbedder and in-memory backends so it
can run in CI, tutorials, or air-gapped notebooks without API keys or model
downloads.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexisearch.chunking import RecursiveChunker
from lexisearch.embeddings import MockEmbedder
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


def build_documents() -> list[Document]:
    """Create a tiny corpus with source metadata for the demo."""
    return [
        Document(
            content=(
                "Platform runbooks should define latency budgets, retries, "
                "and backpressure rules before traffic spikes."
            ),
            metadata=DocumentMetadata(source="runbook.md", title="Reliability runbook"),
        ),
        Document(
            content=(
                "Retrieval augmented generation pipelines need chunking, embeddings, "
                "hybrid search, and source citations."
            ),
            metadata=DocumentMetadata(source="rag-notes.md", title="RAG notes"),
        ),
        Document(
            content=(
                "Evaluation plans compare relevance, answer faithfulness, latency, "
                "and cost before a search release."
            ),
            metadata=DocumentMetadata(source="eval-plan.md", title="Evaluation plan"),
        ),
    ]


def main() -> None:
    """Index the sample corpus and print the top hybrid-search matches."""
    chunker = RecursiveChunker(chunk_size=180, chunk_overlap=0)
    chunks = [chunk for document in build_documents() for chunk in chunker.chunk(document)]

    embedder = MockEmbedder(dimensions=DIMENSIONS)
    embedded_chunks = embedder.embed_chunks(chunks)

    bm25 = BM25Retriever()
    bm25.add_chunks(chunks)

    config = VectorStoreConfig(collection_name="offline-demo", dimensions=DIMENSIONS)
    with InMemoryVectorStore(config=config) as store:
        store.add(embedded_chunks)
        vector = VectorRetriever(store, embedder)
        hybrid = HybridRetriever(
            retrievers=[bm25, vector],
            config=HybridConfig(fusion_method=FusionMethod.LINEAR, weights=[0.85, 0.15], top_k=2),
        )

        query = "latency backpressure runbook"
        results = hybrid.retrieve(query, top_k=2)

    print(f"Query: {query}")
    for result in results:
        title = result.chunk.metadata.get("document_title", "Untitled")
        source = result.chunk.metadata.get("source", "unknown")
        print(f"{result.rank}. {title} ({source}) score={result.score:.4f}")


if __name__ == "__main__":
    main()
