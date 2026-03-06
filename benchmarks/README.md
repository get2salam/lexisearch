# LexiSearch Benchmarks

Performance benchmark suite for the LexiSearch RAG framework.

## Available Benchmarks

| Script | Measures |
|--------|----------|
| `bench_latency.py` | P50/P95/P99 latency for embedding, search, chunking, full pipeline |
| `bench_throughput.py` | docs/sec and queries/sec at single and multi-thread concurrency |

## Quick Start

```bash
# From the repo root
python benchmarks/bench_latency.py
python benchmarks/bench_throughput.py

# Specific suite
python benchmarks/bench_latency.py --suite search --runs 200
python benchmarks/bench_throughput.py --suite retrieval --docs 500 --queries 200
```

## Options

### bench_latency.py

```
--runs N       Number of iterations per benchmark (default: 100)
--suite        embedding | search | chunking | pipeline | all
--verbose      Extra output
```

### bench_throughput.py

```
--docs N       Number of documents to index (default: 200)
--queries N    Number of queries to run (default: 100)
--texts N      Number of texts for embedding throughput (default: 500)
--suite        ingest | retrieval | concurrent | embedding | all
```

## Interpreting Results

### Latency benchmarks

```
  <benchmark name>    mean=X.XXXms  p50=X.XXXms  p95=X.XXXms  p99=X.XXXms
```

- **mean**: average latency across all runs
- **p50**: median latency (50th percentile)
- **p95**: 95th-percentile tail latency — what most users experience in the worst 5% of requests
- **p99**: 99th-percentile tail latency — worst 1% of requests

### Throughput benchmarks

```
  <config>   N ops in X.Xms   →  Y.Y ops/sec
```

## Mock vs Production Backends

All benchmarks run against mock backends by default (no external dependencies
needed).  The mock embedder returns deterministic zero-copy vectors, so
results reflect pure framework overhead rather than model inference time.

For realistic numbers, swap in real backends:

```python
from lexisearch.embeddings import OpenAIEmbedder     # real API
from lexisearch.embeddings import SBERTEmbedder       # local model
from lexisearch.vectorstore import FAISSVectorStore   # FAISS ANN search
```

## Baseline (Mock Backends, April 2025)

These baselines were measured on a standard developer workstation using
the mock embedder and InMemoryVectorStore.  Your mileage will vary.

| Benchmark | Typical Result |
|-----------|----------------|
| MockEmbedder.embed_text() | < 0.01 ms mean |
| InMemoryVectorStore.search() n=1,000 | < 1 ms mean |
| InMemoryVectorStore.search() n=10,000 | < 10 ms mean |
| FixedSizeChunker (2,000 words) | < 2 ms mean |
| Full pipeline query (mock, n=50 docs) | < 5 ms mean |
| Ingest throughput (batch) | > 500 docs/sec |
| Query throughput (single thread) | > 200 queries/sec |

## Adding New Benchmarks

1. Create `benchmarks/bench_<name>.py`
2. Follow the pattern in existing files: `_timed()` for latency, wall-clock
   for throughput
3. Add a `main()` with `argparse` for CLI flags
4. Update this README
