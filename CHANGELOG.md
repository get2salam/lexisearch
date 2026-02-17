# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-17

### Added

- **Vector Store Layer** — Pluggable backends for vector indexing and retrieval
- `BaseVectorStore` ABC with full CRUD: add, upsert, delete, get, search
- `VectorStoreConfig` dataclass for backend-agnostic configuration
- `DistanceMetric` enum: cosine, euclidean, dot product
- `InMemoryVectorStore` — Pure-Python brute-force store with JSON persistence
- `FAISSVectorStore` — Facebook AI Similarity Search with Flat and IVF indexes
- `ChromaVectorStore` — ChromaDB adapter with collection management and HNSW
- `QdrantVectorStore` — Qdrant adapter with payload filtering and HNSW
- Similarity metrics module: `cosine_similarity`, `euclidean_distance`, `dot_product`, `l2_normalize`
- `compute_score` dispatcher and `compute_pairwise_scores` for batch operations
- Context manager support for store lifecycle management
- Metadata filtering on similarity search queries
- 82 new test cases covering metrics, CRUD, search, persistence, and edge cases

### Changed

- Bumped version to 0.2.0
- Updated architecture documentation with vector store layer details
- Updated directory structure in docs

## [0.1.0] - 2026-02-16

### Added

- Core data models: `Document`, `Chunk`, `Embedding`, `SearchResult`, `SearchResponse`
- Document ingestion pipeline with `TextLoader`, `PDFLoader`, `HTMLLoader`
- Chunking strategies: `FixedSizeChunker`, `RecursiveChunker`, `SentenceChunker`, `SemanticChunker`
- Embedding providers: `OpenAIEmbedder`, `SentenceTransformerEmbedder`, `MockEmbedder`
- In-memory embedding cache with batch processing
- Comprehensive test suite (40+ tests)
- CI/CD with GitHub Actions (pytest, ruff, mypy)
- Documentation: architecture guide, getting-started guide, contributing guide
- Pre-commit hooks configuration
- PEP 561 `py.typed` marker for downstream type checking
