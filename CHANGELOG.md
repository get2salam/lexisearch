# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
