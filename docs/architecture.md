# Architecture

LexiSearch follows a modular pipeline architecture where each stage is independently configurable and extensible.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          LexiSearch Pipeline                        │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌─────────────┐  │
│  │          │    │          │    │           │    │             │  │
│  │  Ingest  │───>│ Chunking │───>│ Embedding │───>│Vector Store │  │
│  │          │    │          │    │           │    │             │  │
│  └──────────┘    └──────────┘    └───────────┘    └──────┬──────┘  │
│       │               │               │                  │         │
│       ▼               ▼               ▼                  ▼         │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌─────────────┐  │
│  │ Document │    │  Chunk   │    │ Embedded  │    │   Search    │  │
│  │  Model   │    │  Model   │    │   Chunk   │    │  Response   │  │
│  └──────────┘    └──────────┘    └───────────┘    └─────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     Core Models Layer                        │   │
│  │  Document · DocumentMetadata · Chunk · Embedding · Search   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Models

All data flows through strongly-typed dataclasses defined in `lexisearch/models.py`:

| Model | Purpose |
|-------|---------|
| `Document` | A complete ingested document with content and metadata |
| `DocumentMetadata` | Source, title, author, format, and custom fields |
| `Chunk` | A text segment derived from a document |
| `Embedding` | A dense vector representation of a chunk |
| `EmbeddedChunk` | A chunk paired with its embedding |
| `SearchResult` | A single retrieval result with score and rank |
| `SearchResponse` | A complete query response with multiple results |

## Ingestion Layer

```
                    ┌──────────────┐
                    │  BaseLoader  │  (Abstract)
                    │              │
                    │  + load()    │
                    │  + can_load()│
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌───┴───┐ ┌──────┴──────┐
        │ TextLoader │ │  PDF  │ │  HTMLLoader  │
        │            │ │Loader │ │              │
        │ .txt, .md  │ │ .pdf  │ │ .html, .htm  │
        └────────────┘ └───────┘ └──────────────┘
```

**Key design decisions:**
- Abstract base class pattern for extensibility
- Each loader handles file I/O and format-specific parsing
- Metadata extraction happens at load time
- `load_from_string()` available for in-memory processing

### Adding a Custom Loader

```python
from lexisearch.ingest.base import BaseLoader
from lexisearch.models import Document, DocumentFormat

class DocxLoader(BaseLoader):
    def load(self, source):
        # Your implementation
        ...

    def supported_formats(self):
        return [DocumentFormat.UNKNOWN]  # or add a custom format
```

## Chunking Layer

```
                   ┌──────────────┐
                   │ BaseChunker  │  (Abstract)
                   │              │
                   │ + chunk()    │
                   │ + strategy() │
                   └──────┬───────┘
                          │
         ┌────────────────┼──────────────────┐
         │                │                  │
   ┌─────┴──────┐  ┌─────┴──────┐  ┌────────┴────────┐
   │ FixedSize  │  │ Recursive  │  │    Sentence     │
   │  Chunker   │  │  Chunker   │  │    Chunker      │
   └────────────┘  └────────────┘  └─────────────────┘
                          │
                   ┌──────┴───────┐
                   │   Semantic   │
                   │   Chunker    │
                   └──────────────┘
```

### Strategy Comparison

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| **Fixed-size** | Uniform processing | Simple, predictable | Splits mid-sentence |
| **Recursive** | General use | Respects structure | More complex |
| **Sentence** | Q&A, chat | Clean boundaries | Variable sizes |
| **Semantic** | Topic-based retrieval | Coherent chunks | Requires similarity fn |

### Configurable Parameters

All chunkers accept:
- `chunk_size` — Target chunk size in characters
- `chunk_overlap` — Characters shared between consecutive chunks

## Embedding Layer

```
                   ┌──────────────┐
                   │ BaseEmbedder │  (Abstract)
                   │              │
                   │ + embed_text │
                   │ + embed_batch│
                   │ + embed_chunk│
                   └──────┬───────┘
                          │
         ┌────────────────┼──────────────┐
         │                │              │
   ┌─────┴──────┐  ┌─────┴──────┐  ┌────┴────────┐
   │   OpenAI   │  │   SBERT    │  │    Mock     │
   │  Embedder  │  │  Embedder  │  │  Embedder   │
   │            │  │            │  │  (testing)  │
   └────────────┘  └────────────┘  └─────────────┘
```

**Features:**
- Built-in in-memory caching (opt-in)
- Batch processing for efficiency
- `embed_chunks()` handles cache-aware batching automatically
- Deterministic mock embedder for testing

## Vector Store Layer

```
                   ┌──────────────────┐
                   │  BaseVectorStore │  (Abstract)
                   │                  │
                   │  + add/upsert    │
                   │  + delete/get    │
                   │  + search        │
                   │  + persist/load  │
                   └────────┬─────────┘
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
 ┌─────┴──────┐    ┌───────┴───────┐    ┌───────┴──────┐
 │  InMemory  │    │     FAISS     │    │    Qdrant    │
 │   Store    │    │    Store      │    │    Store     │
 │ (testing)  │    │  (production) │    │ (distributed)│
 └────────────┘    └───────────────┘    └──────────────┘
                           │
                   ┌───────┴──────┐
                   │   ChromaDB   │
                   │    Store     │
                   │ (persistent) │
                   └──────────────┘
```

### Backend Comparison

| Backend | Best For | Index Type | Dependencies |
|---------|----------|------------|--------------|
| **InMemory** | Testing, small datasets | Brute-force | None |
| **FAISS** | Production, large scale | Flat / IVF | `faiss-cpu` |
| **ChromaDB** | Persistence, simplicity | HNSW | `chromadb` |
| **Qdrant** | Distributed, filtering | HNSW | `qdrant-client` |

### Distance Metrics

All backends support three distance metrics via `DistanceMetric`:

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Cosine** | 1 − cos(a,b) | Normalised embeddings (most common) |
| **Euclidean** | √Σ(aᵢ−bᵢ)² | Absolute distance matters |
| **Dot Product** | Σaᵢbᵢ | Pre-normalised or magnitude-aware |

### Usage Pattern

```python
from lexisearch.vectorstore import InMemoryVectorStore, VectorStoreConfig, DistanceMetric

config = VectorStoreConfig(
    collection_name="research_papers",
    dimensions=384,
    metric=DistanceMetric.COSINE,
)

# Use as context manager for automatic lifecycle management
with InMemoryVectorStore(config=config) as store:
    store.add(embedded_chunks)
    results = store.search(query_vector, top_k=5)
    store.persist("my_index.json")
```

### Swapping Backends

```python
# Development / testing
store = InMemoryVectorStore(config=config)

# Production with FAISS
from lexisearch.vectorstore.faiss_store import FAISSVectorStore
store = FAISSVectorStore(config=config, index_type="flat")

# Persistent with ChromaDB
from lexisearch.vectorstore.chroma_store import ChromaVectorStore
store = ChromaVectorStore(config=config, persist_directory="./chroma_data")

# Distributed with Qdrant
from lexisearch.vectorstore.qdrant_store import QdrantVectorStore
store = QdrantVectorStore(config=config, location="http://localhost:6333")
```

## Data Flow Example

```
  research_paper.pdf
        │
        ▼
  ┌─────────────┐
  │  PDFLoader  │   → Document(content="...", metadata=...)
  └──────┬──────┘
         │
         ▼
  ┌──────────────┐
  │  Recursive   │   → [Chunk₁, Chunk₂, ..., Chunkₙ]
  │   Chunker    │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │    OpenAI    │   → [EmbeddedChunk₁, ..., EmbeddedChunkₙ]
  │   Embedder   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Vector Store │   → Index + search via FAISS / Qdrant / ChromaDB
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Search     │   → SearchResponse with ranked SearchResults
  │   Query      │
  └──────────────┘
```

## Directory Structure

```
lexisearch/
├── __init__.py          # Public API exports
├── models.py            # Core data models
├── py.typed             # PEP 561 marker
├── ingest/
│   ├── __init__.py
│   ├── base.py          # BaseLoader ABC
│   ├── text_loader.py   # Plain text / Markdown
│   ├── pdf_loader.py    # PDF via PyMuPDF
│   └── html_loader.py   # HTML via BeautifulSoup
├── chunking/
│   ├── __init__.py
│   ├── base.py          # BaseChunker ABC
│   ├── fixed.py         # Fixed-size chunking
│   ├── recursive.py     # Recursive character splitting
│   ├── sentence.py      # Sentence-boundary chunking
│   └── semantic.py      # Similarity-based chunking
├── embeddings/
│   ├── __init__.py
│   ├── base.py          # BaseEmbedder ABC
│   ├── mock.py          # Deterministic mock
│   ├── openai_embedder.py # OpenAI API
│   └── sbert.py         # Sentence Transformers
└── vectorstore/
    ├── __init__.py       # Public exports
    ├── base.py           # BaseVectorStore ABC + VectorStoreConfig
    ├── metrics.py        # Distance metric functions
    ├── memory.py         # InMemoryVectorStore (brute-force)
    ├── faiss_store.py    # FAISSVectorStore (Flat / IVF)
    ├── chroma_store.py   # ChromaVectorStore (HNSW)
    └── qdrant_store.py   # QdrantVectorStore (HNSW)
```

## Design Principles

1. **Modularity** — Each component is independently replaceable
2. **Type Safety** — Full type annotations with mypy strict mode
3. **Extensibility** — Abstract base classes for all plugin points
4. **Testability** — Mock implementations for offline testing
5. **Production-Ready** — Caching, batch processing, error handling
6. **Backend Agnostic** — Swap vector stores without changing pipeline code
