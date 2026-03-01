"""LexiSearch command-line interface.

Entry point: ``lexisearch`` (registered via pyproject.toml scripts).

Usage::

    lexisearch --help
    lexisearch index path/to/docs/
    lexisearch search "What is retrieval-augmented generation?"
    lexisearch eval samples.jsonl
    lexisearch serve --port 8000
    lexisearch info
"""

from __future__ import annotations

from lexisearch.cli.main import cli

__all__ = ["cli"]
