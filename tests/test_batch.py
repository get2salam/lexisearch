"""Tests for the lexisearch.batch package."""

from __future__ import annotations

import asyncio

import pytest

from lexisearch.batch import (
    BatchQueryProcessor,
    BatchResult,
    QueryJob,
    QueryResult,
    run_batch,
)

# ---------------------------------------------------------------------------
# QueryJob tests
# ---------------------------------------------------------------------------


class TestQueryJob:
    def test_default_priority_zero(self) -> None:
        job = QueryJob("hello")
        assert job.priority == 0

    def test_unique_ids(self) -> None:
        j1 = QueryJob("q")
        j2 = QueryJob("q")
        assert j1.id != j2.id

    def test_metadata_stored(self) -> None:
        job = QueryJob("q", metadata={"user": "alice"})
        assert job.metadata["user"] == "alice"

    def test_repr_truncates_long_query(self) -> None:
        job = QueryJob("x" * 100)
        assert "..." in repr(job)

    def test_repr_short_query(self) -> None:
        job = QueryJob("short")
        assert "short" in repr(job)


# ---------------------------------------------------------------------------
# QueryResult tests
# ---------------------------------------------------------------------------


class TestQueryResult:
    def test_success_true_when_no_error(self) -> None:
        r = QueryResult(job_id="j1", query="q", result="data", latency_ms=10.0)
        assert r.success is True

    def test_success_false_when_error(self) -> None:
        r = QueryResult(job_id="j1", query="q", result=None, latency_ms=5.0, error="oops")
        assert r.success is False

    def test_repr_ok(self) -> None:
        r = QueryResult(job_id="j1", query="q", result="x", latency_ms=12.3)
        assert "ok" in repr(r)

    def test_repr_error(self) -> None:
        r = QueryResult(job_id="j1", query="q", result=None, latency_ms=5.0, error="boom")
        assert "error=" in repr(r)


# ---------------------------------------------------------------------------
# BatchResult tests
# ---------------------------------------------------------------------------


class TestBatchResult:
    def _make_batch(self, successes: int, failures: int) -> BatchResult:
        results = []
        for i in range(successes):
            results.append(QueryResult(f"s{i}", f"q{i}", f"r{i}", 10.0))
        for i in range(failures):
            results.append(QueryResult(f"f{i}", f"q{i}", None, 5.0, error="err"))
        return BatchResult(results=results, total_latency_ms=100.0)

    def test_successful_count(self) -> None:
        batch = self._make_batch(3, 1)
        assert len(batch.successful) == 3

    def test_failed_count(self) -> None:
        batch = self._make_batch(2, 2)
        assert len(batch.failed) == 2

    def test_success_rate(self) -> None:
        batch = self._make_batch(3, 1)
        assert batch.success_rate == pytest.approx(0.75)

    def test_success_rate_empty(self) -> None:
        assert BatchResult().success_rate == 0.0

    def test_avg_latency(self) -> None:
        batch = self._make_batch(2, 0)  # 10ms each
        assert batch.avg_latency_ms == pytest.approx(10.0)

    def test_avg_latency_empty(self) -> None:
        assert BatchResult().avg_latency_ms == 0.0

    def test_len(self) -> None:
        batch = self._make_batch(2, 1)
        assert len(batch) == 3

    def test_repr_contains_stats(self) -> None:
        batch = self._make_batch(4, 1)
        r = repr(batch)
        assert "success=4" in r
        assert "failed=1" in r
        assert "total=5" in r


# ---------------------------------------------------------------------------
# BatchQueryProcessor tests (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBatchQueryProcessor:
    async def test_basic_batch(self) -> None:
        async def fn(q: str) -> str:
            return f"result:{q}"

        processor = BatchQueryProcessor(fn)
        jobs = [QueryJob("q1"), QueryJob("q2"), QueryJob("q3")]
        batch = await processor.process(jobs)
        assert len(batch) == 3
        assert batch.success_rate == 1.0
        results_set = {r.result for r in batch.results}
        assert results_set == {"result:q1", "result:q2", "result:q3"}

    async def test_empty_batch(self) -> None:
        async def fn(q: str) -> str:
            return q

        processor = BatchQueryProcessor(fn)
        batch = await processor.process([])
        assert len(batch) == 0

    async def test_error_isolation(self) -> None:
        async def fn(q: str) -> str:
            if q == "bad":
                raise ValueError("intentional failure")
            return f"ok:{q}"

        processor = BatchQueryProcessor(fn)
        batch = await processor.process([QueryJob("good"), QueryJob("bad"), QueryJob("good2")])
        assert len(batch.successful) == 2
        assert len(batch.failed) == 1
        assert "intentional failure" in batch.failed[0].error  # type: ignore[arg-type]

    async def test_timeout_triggers(self) -> None:
        async def slow_fn(q: str) -> str:
            await asyncio.sleep(10)
            return q

        processor = BatchQueryProcessor(slow_fn, timeout_seconds=0.05)
        batch = await processor.process([QueryJob("q")])
        assert batch.failed[0].error is not None
        assert "Timed out" in batch.failed[0].error  # type: ignore[arg-type]

    async def test_no_timeout_when_zero(self) -> None:
        async def fn(q: str) -> str:
            return q

        processor = BatchQueryProcessor(fn, timeout_seconds=0)
        batch = await processor.process([QueryJob("q")])
        assert batch.success_rate == 1.0

    async def test_priority_ordering(self) -> None:
        """Higher-priority jobs should be dispatched first."""
        order: list[str] = []

        async def fn(q: str) -> str:
            order.append(q)
            return q

        # Semaphore=1 so jobs run sequentially in priority order
        processor = BatchQueryProcessor(fn, max_concurrent=1)
        jobs = [
            QueryJob("low", priority=1),
            QueryJob("high", priority=10),
            QueryJob("medium", priority=5),
        ]
        await processor.process(jobs)
        assert order[0] == "high"
        assert order[-1] == "low"

    async def test_concurrency_limited_by_semaphore(self) -> None:
        active: list[int] = [0]
        peak: list[int] = [0]

        async def fn(q: str) -> str:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1
            return q

        processor = BatchQueryProcessor(fn, max_concurrent=3)
        jobs = [QueryJob(f"q{i}") for i in range(10)]
        await processor.process(jobs)
        assert peak[0] <= 3

    async def test_latency_recorded(self) -> None:
        async def fn(q: str) -> str:
            return q

        processor = BatchQueryProcessor(fn)
        batch = await processor.process([QueryJob("q")])
        assert batch.results[0].latency_ms >= 0

    async def test_metadata_forwarded(self) -> None:
        async def fn(q: str) -> str:
            return q

        processor = BatchQueryProcessor(fn)
        job = QueryJob("q", metadata={"user": "alice"})
        batch = await processor.process([job])
        assert batch.results[0].metadata["user"] == "alice"

    async def test_process_simple(self) -> None:
        async def fn(q: str) -> str:
            return f"done:{q}"

        processor = BatchQueryProcessor(fn)
        batch = await processor.process_simple(["a", "b", "c"])
        assert len(batch) == 3
        assert all(r.success for r in batch.results)


# ---------------------------------------------------------------------------
# run_batch (synchronous wrapper) tests
# ---------------------------------------------------------------------------


class TestRunBatch:
    def test_run_batch_basic(self) -> None:
        def sync_fn(q: str) -> str:
            return f"result:{q}"

        batch = run_batch(sync_fn, ["q1", "q2"])
        assert len(batch) == 2
        assert batch.success_rate == 1.0

    def test_run_batch_error_isolation(self) -> None:
        def sync_fn(q: str) -> str:
            if q == "fail":
                raise RuntimeError("boom")
            return q

        batch = run_batch(sync_fn, ["ok", "fail"])
        assert len(batch.successful) == 1
        assert len(batch.failed) == 1

    def test_run_batch_empty(self) -> None:
        def sync_fn(q: str) -> str:
            return q

        batch = run_batch(sync_fn, [])
        assert len(batch) == 0

    def test_run_batch_concurrency(self) -> None:
        import threading

        peak = [0]
        active = [0]
        lock = threading.Lock()

        def sync_fn(q: str) -> str:
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            import time

            time.sleep(0.01)
            with lock:
                active[0] -= 1
            return q

        run_batch(sync_fn, [f"q{i}" for i in range(10)], max_concurrent=3)
        assert peak[0] <= 3
