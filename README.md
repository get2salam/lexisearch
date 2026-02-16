<p align="center">
  <h1 align="center">🔍 LexiSearch</h1>
  <p align="center">
    <strong>A production-ready RAG framework for intelligent document search and retrieval</strong>
  </p>
  <p align="center">
    <a href="https://github.com/get2salam/lexisearch/actions/workflows/ci.yml"><img src="https://github.com/get2salam/lexisearch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://pypi.org/project/lexisearch/"><img src="https://img.shields.io/pypi/v/lexisearch.svg" alt="PyPI"></a>
    <a href="https://pypi.org/project/lexisearch/"><img src="https://img.shields.io/pypi/pyversions/lexisearch.svg" alt="Python"></a>
    <a href="https://github.com/get2salam/lexisearch/blob/main/LICENSE"><img src="https://img.shields.io/github/license/get2salam/lexisearch.svg" alt="License"></a>
    <a href="https://github.com/get2salam/lexisearch"><img src="https://img.shields.io/github/stars/get2salam/lexisearch.svg?style=social" alt="Stars"></a>
  </p>
</p>

---

**LexiSearch** is a modular, extensible Retrieval-Augmented Generation (RAG) framework built for production workloads. It provides a clean pipeline for ingesting documents, chunking text intelligently, generating embeddings, and retrieving relevant content for LLM-powered applications.

## ✨ Features

- **📄 Multi-format Ingestion** — PDF, HTML, plain text, with extensible loader architecture
- **✂️ Smart Chunking** — Fixed-size, recursive, semantic, and sentence-based strategies with configurable overlap
- **🧠 Flexible Embeddings** — OpenAI, Sentence Transformers, or bring your own
- **⚡ Token-Aware** — Built-in tiktoken integration for precise chunk sizing
- **🔌 Extensible** — Abstract base classes for every component; plug in your own implementations
- **📊 Type-Safe** — Fully typed with mypy strict mode, dataclass-based models
- **🧪 Well-Tested** — Comprehensive test suite with 80%+ coverage target
- **🚀 Async-Ready** — Async interfaces for high-throughput pipelines

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        LexiSearch                           │
├──────────┬──────────┬──────────────┬───────────────────────-┤
│  Ingest  │ Chunking │  Embeddings  │       Retrieval        │
│          │          │              │    (coming soon)        │
│ ┌──────┐ │ ┌──────┐ │  ┌────────┐  │                        │
│ │ PDF  │ │ │Fixed │ │  │ OpenAI │  │  ┌──────────────────┐  │
│ │ HTML │ │ │Recur.│ │  │  SBERT │  │  │  Vector Stores   │  │
│ │ Text │ │ │Sent. │ │  │ Custom │  │  │  (coming soon)   │  │
│ │Custom│ │ │Seman.│ │  │        │  │  └──────────────────┘  │
│ └──────┘ │ └──────┘ │  └────────┘  │                        │
├──────────┴──────────┴──────────────┴────────────────────────┤
│                    Core Models & Types                       │
│            Document · Chunk · Embedding · SearchResult       │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Core package
pip install lexisearch

# With PDF support
pip install lexisearch[pdf]

# With all optional dependencies
pip install lexisearch[all]

# For development
pip install lexisearch[dev]
```

### From Source

```bash
git clone https://github.com/get2salam/lexisearch.git
cd lexisearch
pip install -e ".[dev]"
```

## 🚀 Quickstart

### Ingest a Document

```python
from lexisearch.ingest import TextLoader

loader = TextLoader()
documents = loader.load("path/to/document.txt")
```

### Chunk Text

```python
from lexisearch.chunking import RecursiveChunker

chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(documents[0])
```

### Generate Embeddings

```python
from lexisearch.embeddings import MockEmbedder

embedder = MockEmbedder(dimensions=384)
embedded_chunks = embedder.embed_chunks(chunks)
```

### Full Pipeline

```python
from lexisearch.ingest import TextLoader
from lexisearch.chunking import RecursiveChunker
from lexisearch.embeddings import MockEmbedder

# Load
loader = TextLoader()
docs = loader.load("research_paper.txt")

# Chunk
chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
all_chunks = []
for doc in docs:
    all_chunks.extend(chunker.chunk(doc))

# Embed
embedder = MockEmbedder(dimensions=384)
embedded = embedder.embed_chunks(all_chunks)

print(f"Processed {len(docs)} documents into {len(embedded)} embedded chunks")
```

## 📖 Documentation

- [Getting Started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Contributing](CONTRIBUTING.md)

## 🗺️ Roadmap

- [x] Core models and types
- [x] Document ingestion (PDF, HTML, Text)
- [x] Chunking strategies (fixed, recursive, semantic, sentence)
- [x] Embedding layer (OpenAI, Sentence Transformers)
- [ ] Vector store integrations (FAISS, ChromaDB, Pinecone)
- [ ] Retrieval pipeline with reranking
- [ ] Query expansion and HyDE
- [ ] Evaluation framework (RAGAS metrics)
- [ ] REST API server
- [ ] Web UI for document management

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with inspiration from the RAG research community and production search systems.
