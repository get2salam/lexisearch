"""Async batch query processor.

:class:`BatchQueryProcessor` executes multiple RAG queries concurrently,
isolating per-query errors and collecting results with timing metadata.
The synchronous :func:`run_batch` wrapper is provided for calling code that
does not manage an event loop.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


@dataclass
class QueryJob:
    """A single query job for batch execution.

    Attributes:
        query: The query string to execute.
        id: Unique job identifier.
        priority: Higher value means the job is scheduled first.
        metadata: Arbitrary job-level metadata (e.g. user ID, session ID).
    """

    query: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        preview = self.query[:40] + "..." if len(self.query) > 40 else self.query
        return f"QueryJob(id={self.id!r}, priority={self.priority}, query={preview!r})"


@dataclass
class QueryResult:
    """Result of a single batch query job.

    Attributes:
        job_id: ID of the originating :class:`QueryJob`.
        query: The original query string.
        result: The return value of the query function, or ``None`` on error.
        latency_ms: Wall-clock time in milliseconds.
        error: Error message string if the job failed, otherwise ``None``.
        metadata: Job-level metadata forwarded from the input :class:`QueryJob`.
    """

    job_id: str
    query: str
    result: Any
    latency_ms: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Return ``True`` if the job completed without error.

        Returns:
            Job success flag.
        """
        return self.error is None

    def __repr__(self) -> str:
        """Return a concise string representation."""
        status = "ok" if self.success else f"error={self.error!r}"
        return f"QueryResult(job_id={self.job_id!r}, latency={self.latency_ms:.1f}ms, {status})"


@dataclass
class BatchResult:
    """Aggregated results from a completed batch.

    Attributes:
        results: Ordered list of per-job results.
        total_latency_ms: Wall-clock time for the entire batch.
    """

    results: list[QueryResult] = field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def successful(self) -> list[QueryResult]:
        """Return only successful job results.

        Returns:
            Subset of results with no error.
        """
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> list[QueryResult]:
        """Return only failed job results.

        Returns:
            Subset of results with an error.
        """
        return [r for r in self.results if not r.success]

    @property
    def success_rate(self) -> float:
        """Fraction of jobs that succeeded.

        Returns:
            Value in ``[0.0, 1.0]``.
        """
        if not self.results:
            return 0.0
        return len(self.successful) / len(self.results)

    @property
    def avg_latency_ms(self) -> float:
        """Mean latency across all jobs.

        Returns:
            Average latency in milliseconds, or ``0.0`` if no results.
        """
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    def __len__(self) -> int:
        """Return total number of results."""
        return len(self.results)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"BatchResult("
            f"total={len(self)}, "
            f"success={len(self.successful)}, "
            f"failed={len(self.failed)}, "
            f"success_rate={self.success_rate:.0%}, "
            f"total_latency={self.total_latency_ms:.1f}ms)"
        )


class BatchQueryProcessor:
    """Async batch processor for concurrent RAG queries.

    Processes multiple :class:`QueryJob` objects concurrently, bounded by a
    semaphore.  Each job is individually isolated — a timeout or exception in
    one job never cancels others.  Jobs are dispatched in priority order
    (highest priority first).

    Args:
        query_fn: Async callable that accepts a query string and returns a
            result.
        max_concurrent: Maximum number of simultaneous in-flight queries.
        timeout_seconds: Per-query timeout in seconds.  ``0`` disables the
            timeout.

    Example:
        >>> async def my_retriever(query: str) -> list[str]:
        ...     return ["result"]
        >>> processor = BatchQueryProcessor(my_retriever, max_concurrent=4)
        >>> batch = await processor.process([QueryJob("query 1"), QueryJob("query 2")])
        >>> batch.success_rate
        1.0
    """

    def __init__(
        self,
        query_fn: Callable[[str], Coroutine[Any, Any, Any]],
        max_concurrent: int = 5,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialise the BatchQueryProcessor.

        Args:
            query_fn: Async query function.
            max_concurrent: Concurrency limit.
            timeout_seconds: Per-query timeout.  0 = no timeout.
        """
        self.query_fn = query_fn
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def process(self, jobs: list[QueryJob]) -> BatchResult:
        """Process a list of query jobs concurrently.

        Jobs are sorted by :attr:`QueryJob.priority` (highest first) before
        dispatch.  All jobs are awaited concurrently up to ``max_concurrent``
        simultaneous executions.

        Args:
            jobs: Query jobs to process.

        Returns:
            :class:`BatchResult` with per-job outcomes and aggregate stats.
        """
        if not jobs:
            return BatchResult()

        sorted_jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)
        start = time.perf_counter()
        results: list[QueryResult] = list(
            await asyncio.gather(*[self._run_job(job) for job in sorted_jobs])
        )
        total_ms = (time.perf_counter() - start) * 1000
        return BatchResult(results=results, total_latency_ms=total_ms)

    async def process_simple(self, queries: list[str]) -> BatchResult:
        """Convenience wrapper for a plain list of query strings.

        Args:
            queries: Query strings to process.

        Returns:
            :class:`BatchResult` with per-query outcomes.
        """
        return await self.process([QueryJob(query=q) for q in queries])

    async def _run_job(self, job: QueryJob) -> QueryResult:
        """Execute a single job under the concurrency semaphore.

        Args:
            job: The job to execute.

        Returns:
            :class:`QueryResult` — always returns, never raises.
        """
        async with self._semaphore:
            start = time.perf_counter()
            try:
                if self.timeout_seconds > 0:
                    result = await asyncio.wait_for(
                        self.query_fn(job.query),
                        timeout=self.timeout_seconds,
                    )
                else:
                    result = await self.query_fn(job.query)
                latency = (time.perf_counter() - start) * 1000
                return QueryResult(
                    job_id=job.id,
                    query=job.query,
                    result=result,
                    latency_ms=latency,
                    metadata=dict(job.metadata),
                )
            except asyncio.TimeoutError:
                latency = (time.perf_counter() - start) * 1000
                return QueryResult(
                    job_id=job.id,
                    query=job.query,
                    result=None,
                    latency_ms=latency,
                    error=f"Timed out after {self.timeout_seconds}s",
                    metadata=dict(job.metadata),
                )
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000
                return QueryResult(
                    job_id=job.id,
                    query=job.query,
                    result=None,
                    latency_ms=latency,
                    error=str(exc),
                    metadata=dict(job.metadata),
                )


def run_batch(
    query_fn: Callable[[str], Any],
    queries: list[str],
    max_concurrent: int = 5,
    timeout_seconds: float = 30.0,
) -> BatchResult:
    """Synchronous convenience wrapper for batch query processing.

    Wraps a **synchronous** ``query_fn`` in :func:`asyncio.to_thread` and
    runs the batch using :func:`asyncio.run`.  Do not call this inside an
    existing event loop; use :class:`BatchQueryProcessor` directly in async
    code instead.

    Args:
        query_fn: Synchronous function ``(query: str) -> Any``.
        queries: List of query strings.
        max_concurrent: Maximum concurrent workers.
        timeout_seconds: Per-query timeout.

    Returns:
        :class:`BatchResult` with outcomes for all queries.

    Example:
        >>> results = run_batch(my_retriever, ["q1", "q2", "q3"], max_concurrent=3)
        >>> results.success_rate
        1.0
    """

    async def _async_fn(query: str) -> Any:
        return await asyncio.to_thread(query_fn, query)

    processor = BatchQueryProcessor(
        query_fn=_async_fn,
        max_concurrent=max_concurrent,
        timeout_seconds=timeout_seconds,
    )
    return asyncio.run(processor.process_simple(queries))
