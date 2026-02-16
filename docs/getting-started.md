# Getting Started

This guide walks you through installing LexiSearch and running your first document processing pipeline.

## Installation

### Basic Install

```bash
pip install lexisearch
```

### With Optional Dependencies

```bash
# PDF support
pip install lexisearch[pdf]

# HTML support
pip install lexisearch[html]

# OpenAI embeddings
pip install lexisearch[openai]

# Sentence Transformers (local embeddings)
pip install lexisearch[sentence-transformers]

# Everything
pip install lexisearch[all]
```

### From Source (Development)

```bash
git clone https://github.com/get2salam/lexisearch.git
cd lexisearch
pip install -e ".[dev]"
```

## Quick Example

### 1. Load a Document

```python
from lexisearch.ingest import TextLoader

loader = TextLoader()
documents = loader.load("path/to/your/document.txt")

print(f"Loaded {len(documents)} document(s)")
print(f"Content length: {documents[0].char_count} characters")
print(f"Word count: {documents[0].word_count}")
```

### 2. Chunk the Document

```python
from lexisearch.chunking import RecursiveChunker

chunker = RecursiveChunker(
    chunk_size=512,     # Target characters per chunk
    chunk_overlap=50,   # Overlap between consecutive chunks
)

chunks = chunker.chunk(documents[0])
print(f"Created {len(chunks)} chunks")

for chunk in chunks[:3]:
    print(f"  Chunk {chunk.index}: {chunk.char_count} chars")
```

### 3. Generate Embeddings

```python
from lexisearch.embeddings import MockEmbedder

# Use MockEmbedder for testing (no API key needed)
embedder = MockEmbedder(dimensions=384)

embedded_chunks = embedder.embed_chunks(chunks)
print(f"Embedded {len(embedded_chunks)} chunks")
print(f"Vector dimensions: {embedded_chunks[0].embedding.dimensions}")
```

### 4. Full Pipeline

```python
from lexisearch.ingest import TextLoader
from lexisearch.chunking import RecursiveChunker
from lexisearch.embeddings import MockEmbedder

# Configure
loader = TextLoader()
chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
embedder = MockEmbedder(dimensions=384)

# Process
docs = loader.load("research_paper.txt")
all_chunks = chunker.chunk_many(docs)
embedded = embedder.embed_chunks(all_chunks)

print(f"Pipeline complete:")
print(f"  Documents: {len(docs)}")
print(f"  Chunks: {len(all_chunks)}")
print(f"  Embedded: {len(embedded)}")
```

## Choosing a Chunking Strategy

| Strategy | Use Case | Import |
|----------|----------|--------|
| `FixedSizeChunker` | Simple, uniform chunks | `from lexisearch.chunking import FixedSizeChunker` |
| `RecursiveChunker` | General purpose (recommended) | `from lexisearch.chunking import RecursiveChunker` |
| `SentenceChunker` | Q&A, chatbots | `from lexisearch.chunking import SentenceChunker` |
| `SemanticChunker` | Topic-aware splitting | `from lexisearch.chunking import SemanticChunker` |

## Choosing an Embedding Provider

### Mock (Testing)

```python
from lexisearch.embeddings import MockEmbedder
embedder = MockEmbedder(dimensions=384)
```

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

```python
from lexisearch.embeddings import OpenAIEmbedder
embedder = OpenAIEmbedder(model="text-embedding-3-small")
```

### Sentence Transformers (Local)

```python
from lexisearch.embeddings import SentenceTransformerEmbedder
embedder = SentenceTransformerEmbedder(model_name_or_path="all-MiniLM-L6-v2")
```

## Extending LexiSearch

### Custom Loader

```python
from lexisearch.ingest.base import BaseLoader
from lexisearch.models import Document, DocumentFormat

class CSVLoader(BaseLoader):
    def load(self, source):
        # Your CSV parsing logic here
        ...

    def supported_formats(self):
        return [DocumentFormat.UNKNOWN]
```

### Custom Chunker

```python
from lexisearch.chunking.base import BaseChunker
from lexisearch.models import Chunk, ChunkStrategy, Document

class ParagraphChunker(BaseChunker):
    def chunk(self, document: Document) -> list[Chunk]:
        paragraphs = document.content.split("\n\n")
        # Build chunks from paragraphs
        ...

    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.RECURSIVE
```

## Next Steps

- Read the [Architecture Guide](architecture.md) for system design details
- Check out the [examples/](../examples/) directory for more usage patterns
- See [CONTRIBUTING.md](../CONTRIBUTING.md) to contribute
