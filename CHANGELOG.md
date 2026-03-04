# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-03-04

### Fixed

- **Type Safety** — Resolved 41 mypy type checking errors across API routes and CLI
- Removed 12 unused `type: ignore` comments (routes: evaluate, query, health, documents, server)
- Fixed `list.append()` → `list.extend()` type mismatch in CLI document loader
- Added mypy override for CLI module to allow untyped click decorators
- Removed unused `type: ignore[union-attr]` in plugins registry

### Changed

- **CI/CD** — All type checking, linting, and tests now passing on Python 3.10-3.13

## [0.3.0] - 2026-02-19

### Added

- **Retrieval Engine** — Hybrid search, reranking, diversity, and query expansion
- `BaseRetriever` ABC with configurable scoring, filtering, and timed search
- `MetadataFilter` with 9 operators (eq, neq, gt, gte, lt, lte, in, not_in, contains)
- `BM25Retriever` — Okapi BM25 sparse retriever with inverted index and configurable tokenisation
- `VectorRetriever` — Dense retriever adapter wrapping any `BaseVectorStore`
- `HybridRetriever` — Multi-retriever fusion with RRF, linear weighted, and DBSF strategies
- `RerankedRetriever` — Two-stage retrieve-then-rerank pipeline pattern
- `CrossEncoderReranker` — Sentence-Transformers cross-encoder reranking (lazy-loaded)
- `CohereReranker` — Cohere Rerank API integration
- `LinearScoreReranker` — Lightweight feature-based reranking (term coverage, exact match)
- `mmr_select()` — Maximal Marginal Relevance for relevance-diversity trade-off
- `greedy_diversify()` — Near-duplicate removal via similarity threshold
- `SynonymExpander` — Dictionary-based query term expansion
- `QueryDecomposer` — Split complex queries into focused sub-queries
- `PseudoRelevanceFeedback` — Rocchio-style blind feedback from top results
- `MultiQueryExpander` — Generate keyword, reversed, and declarative query variants
- 91 new test cases covering all retrieval components and integration pipelines

### Changed

- Bumped version to 0.3.0
- Updated architecture documentation with retrieval engine layer
- Updated README roadmap and feature list

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
