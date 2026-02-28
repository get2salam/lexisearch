# LexiSearch — Claude Code Project Instructions

## What This Is
LexiSearch is a **RAG (Retrieval-Augmented Generation) framework for legal research**. It provides hybrid search (semantic + keyword), citation extraction, and AI-powered legal analysis. Generic design — works with any legal jurisdiction.

**GitHub**: `get2salam/lexisearch` (PUBLIC — no Pakistan/PLS references ever)

## Tech Stack
- **Python 3.11+** with type hints everywhere
- **FastAPI** backend with async endpoints
- **ChromaDB** for vector storage
- **Sentence-transformers** for embeddings
- **Ruff** for linting + formatting (config in pyproject.toml)
- **Pytest** for testing (246+ tests)

## Project Structure
```
src/lexisearch/
├── core/           # Core search, indexing, embedding logic
├── api/            # FastAPI routes and middleware
├── models/         # Pydantic models and schemas
├── services/       # Business logic services
├── utils/          # Helpers, caching, text processing
└── config/         # Settings and configuration
tests/              # Pytest test suite (mirrors src structure)
```

## Key Patterns
- All services use dependency injection via FastAPI's `Depends()`
- Embeddings go through `embed_text_cached()` (NOT `embed_text()` directly — cache bypass bug)
- Config via pydantic-settings with `.env` file support
- ClassVar fields need `ClassVar[type]` annotation (ruff RUF012)
- Exception chaining: always use `raise X from err` (ruff B904)

## Linting Rules (pyproject.toml)
- Ruff rules: E, F, W, I, N, D, B, UP, RUF, S
- Test files exempt from D102, D103, D105, D107 (docstrings)
- Line length: 120 chars
- Target: Python 3.11

## Before Committing
1. `ruff check src/ tests/` — must be 0 issues
2. `ruff format src/ tests/` — must have no changes
3. `pytest` — all tests must pass
4. No `print()` in production code (use logging)
5. No Pakistan/PLS/Qanoon references — this repo is PUBLIC

## Common Gotchas
- `embed_text()` bypasses cache → always use `embed_text_cached()`
- `str.isupper()` returns False for mixed-case like "PCrLJ" → use explicit lists
- ChromaDB Settings import is unused after refactor → don't re-add it (F401)
