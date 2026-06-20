"""Tests for runnable documentation examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_offline_hybrid_search_example_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example = repo_root / "examples" / "offline_hybrid_search.py"

    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Query: latency backpressure runbook" in result.stdout
    assert "1. Reliability runbook (runbook.md)" in result.stdout
    assert "score=" in result.stdout
