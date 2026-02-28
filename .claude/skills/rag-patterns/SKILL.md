---
name: rag-patterns
description: RAG (Retrieval-Augmented Generation) patterns and conventions for the LexiSearch codebase. Use when implementing search, indexing, embedding, or retrieval features.
---

# LexiSearch RAG Patterns

## Embedding
- Always use `embed_text_cached()` — NEVER call `embed_text()` directly (bypasses cache)
- Default model: sentence-transformers (configurable via settings)
- Embeddings are cached in-memory with LRU eviction

## Search Pipeline
1. **Query preprocessing** — normalize, extract entities, detect intent
2. **Hybrid retrieval** — combine semantic (vector) + keyword (BM25) results
3. **Re-ranking** — score and sort combined results
4. **Citation extraction** — link results to source documents
5. **Response generation** — synthesize answer with citations

## Indexing
- Documents split into chunks with configurable overlap
- Each chunk gets: embedding, metadata (source, page, section), unique ID
- ChromaDB as vector store (may migrate to pgvector)
- JSONL as interchange format for bulk operations

## Code Patterns
- Services use dependency injection via FastAPI `Depends()`
- All public methods need type hints and docstrings
- Async preferred for I/O-bound operations
- Config via pydantic-settings BaseSettings

## Testing
- 246+ tests, must all pass before commit
- Test files exempt from docstring requirements (D102, D103, D105, D107)
- Use pytest fixtures for shared setup
- Mock external services (LLM, vector DB) in unit tests
