"""Async batch query processing for LexiSearch RAG pipelines.

This package provides concurrency-bounded batch execution of RAG queries,
with per-job error isolation, priority scheduling, and aggregate metrics.

Components:

- :class:`~lexisearch.batch.processor.QueryJob` - a single query job with priority.
- :class:`~lexisearch.batch.processor.QueryResult` - outcome of one job.
- :class:`~lexisearch.batch.processor.BatchResult` - aggregated batch outcome.
- :class:`~lexisearch.batch.processor.BatchQueryProcessor` - async processor.
- :func:`~lexisearch.batch.processor.run_batch` - synchronous convenience wrapper.

Example:
    >>> import asyncio
    >>> from lexisearch.batch import BatchQueryProcessor, QueryJob
    >>>
    >>> async def search(query: str) -> list[str]:
    ...     return [f"result for {query}"]
    >>>
    >>> processor = BatchQueryProcessor(search, max_concurrent=4)
    >>> jobs = [QueryJob(q) for q in ["q1", "q2", "q3"]]
    >>> batch = asyncio.run(processor.process(jobs))
    >>> batch.success_rate
    1.0
"""

from __future__ import annotations

from lexisearch.batch.processor import (
    BatchQueryProcessor,
    BatchResult,
    QueryJob,
    QueryResult,
    run_batch,
)

__all__ = [
    "BatchQueryProcessor",
    "BatchResult",
    "QueryJob",
    "QueryResult",
    "run_batch",
]
