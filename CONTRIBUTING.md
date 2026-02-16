# Contributing to LexiSearch

Thank you for your interest in contributing to LexiSearch! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10 or later
- Git

### Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/get2salam/lexisearch.git
cd lexisearch

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with all development dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

### Verify Your Setup

```bash
# Run tests
make test

# Run linter
make lint

# Run type checker
make typecheck

# Run all checks
make all
```

## Making Changes

### Branch Naming

- `feat/description` — New features
- `fix/description` — Bug fixes
- `docs/description` — Documentation updates
- `refactor/description` — Code refactoring
- `test/description` — Test additions or changes

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add FAISS vector store integration
fix: handle empty documents in recursive chunker
docs: update getting-started guide with new examples
test: add parametrized tests for embedding cache
refactor: simplify chunk metadata building
```

### Code Style

- **Type hints** on all function signatures (mypy strict)
- **Docstrings** on all public methods (Google style)
- **Ruff** for linting and formatting
- **Line length** maximum: 100 characters

Example:

```python
def embed_text(self, text: str) -> list[float]:
    """Generate an embedding vector for the input text.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        ValueError: If the text is empty.
    """
    ...
```

## Pull Request Process

1. **Fork** the repository and create a feature branch
2. **Write tests** for your changes
3. **Run all checks**: `make all`
4. **Open a PR** against `main` with a clear description
5. **Address review feedback** promptly

### PR Checklist

- [ ] Tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking passes (`make typecheck`)
- [ ] New code has docstrings and type hints
- [ ] Test coverage is maintained or improved
- [ ] Documentation updated if applicable

## Adding a New Component

### New Loader

1. Create `lexisearch/ingest/your_loader.py`
2. Inherit from `BaseLoader`
3. Implement `load()` and `supported_formats()`
4. Add to `lexisearch/ingest/__init__.py`
5. Write tests in `tests/test_ingest.py`

### New Chunker

1. Create `lexisearch/chunking/your_chunker.py`
2. Inherit from `BaseChunker`
3. Implement `chunk()` and `strategy()`
4. Add to `lexisearch/chunking/__init__.py`
5. Write tests in `tests/test_chunking.py`

### New Embedder

1. Create `lexisearch/embeddings/your_embedder.py`
2. Inherit from `BaseEmbedder`
3. Implement `embed_text()`, `embed_batch()`, `model_name()`, `dimensions()`
4. Add to `lexisearch/embeddings/__init__.py`
5. Write tests in `tests/test_embeddings.py`

## Reporting Issues

When filing an issue, please include:

- Python version (`python --version`)
- LexiSearch version (`pip show lexisearch`)
- Minimal reproducible example
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
