"""Adaptive retrieval with dynamic k-selection and coverage estimation.

Standard retrieval fixes ``top_k`` upfront, which is a trade-off: too small
misses relevant documents; too large adds noise.  This module provides an
``AdaptiveRetriever`` that adjusts k dynamically:

1. Start with a small initial k (e.g. 3).
2. Measure *coverage* — what fraction of meaningful query terms appear in
   the retrieved chunk set.
3. If coverage is below ``min_coverage``, double k (up to ``max_k``) and
   retrieve again.
4. Return results at the k where coverage stabilises or max_k is reached.

The coverage metric is intentionally simple (keyword-based) so it adds
negligible latency.  It correlates well with recall in practice because
queries whose key terms are not covered by retrieved chunks will almost
certainly fail downstream.

Additional behaviour
--------------------
* Score diversity check: if all top chunks have nearly identical scores
  (within ``score_spread_threshold``), the retriever interprets this as
  "the index doesn't discriminate for this query" and returns early.
* Expansion history is recorded in ``AdvancedRetrievalResult.metadata``
  under the key ``"adaptive"``.

Example::

    from lexisearch.retrieval.adaptive import AdaptiveRetriever, AdaptiveConfig

    retriever = AdaptiveRetriever(
        base_retriever=my_base_retriever,
        config=AdaptiveConfig(initial_k=3, max_k=20, min_coverage=0.6),
    )
    result = retriever.retrieve("What is the doctrine of promissory estoppel?")
    print(result.metadata["adaptive"])
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "of",
        "in",
        "to",
        "for",
        "on",
        "at",
        "with",
        "by",
        "from",
        "and",
        "or",
        "but",
        "if",
        "that",
        "this",
        "it",
        "its",
        "not",
        "no",
        "as",
        "so",
    }
)


def _query_keywords(query: str) -> set[str]:
    """Extract meaningful keywords from a query string."""
    return {t for t in re.findall(r"\b[a-z]{2,}\b", query.lower()) if t not in _STOP_WORDS}


def _chunks_coverage(chunks: list[Any], query_kws: set[str]) -> float:
    """Fraction of *query_kws* that appear in any of the retrieved *chunks*.

    Returns 1.0 when query_kws is empty (vacuously covered).
    """
    if not query_kws:
        return 1.0
    all_terms: set[str] = set()
    for chunk in chunks:
        content = getattr(chunk, "content", "") or ""
        all_terms.update(re.findall(r"\b[a-z]{2,}\b", content.lower()))
    covered = query_kws & all_terms
    return len(covered) / len(query_kws)


def _score_spread(chunks: list[Any]) -> float:
    """Return max-score minus min-score for a list of chunks.

    Used to detect uniform-score results where increasing k won't help.
    """
    if len(chunks) < 2:
        return 0.0
    scores = [float(getattr(c, "score", 0.0)) for c in chunks]
    return max(scores) - min(scores)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveConfig:
    """Configuration for the adaptive retriever.

    Attributes:
        initial_k: Starting number of results to retrieve.
        max_k: Hard upper bound on the number of results.
        min_coverage: Minimum acceptable coverage to stop expansion.
            Range [0.0, 1.0].  0.0 means never expand (always return
            initial_k results). 1.0 means expand until all query terms
            are covered or max_k is reached.
        expand_factor: Multiplier applied to k at each expansion step.
            Default 2 (doubles k each round).
        score_spread_threshold: If score spread < this value, skip
            further expansion (index can't discriminate the query).
    """

    initial_k: int = 3
    max_k: int = 20
    min_coverage: float = 0.6
    expand_factor: float = 2.0
    score_spread_threshold: float = 0.01


# ---------------------------------------------------------------------------
# Expansion record
# ---------------------------------------------------------------------------


@dataclass
class ExpansionStep:
    """One step in the adaptive expansion history."""

    k: int
    coverage: float
    num_chunks: int
    score_spread: float
    reason: str
    """Human-readable reason for expanding (or stopping) at this step."""


# ---------------------------------------------------------------------------
# AdaptiveRetriever
# ---------------------------------------------------------------------------


class AdaptiveRetriever:
    """Adaptively expand k until query coverage is sufficient.

    Parameters
    ----------
    base_retriever:
        Callable ``(query: str, top_k: int) -> list[RetrievedChunk]`` or
        an object with a ``.retrieve(query, top_k)`` method.
    config:
        :class:`AdaptiveConfig` controlling expansion behaviour.

    Notes:
    -----
    The base_retriever is called with increasing k values.  Most vector
    stores support ``top_k`` natively, so each call is a fresh lookup.
    Results from the largest k are always returned (no merging across
    expansion steps — that would change ranking semantics).
    """

    def __init__(
        self,
        base_retriever: Any,
        config: AdaptiveConfig | None = None,
    ) -> None:
        """Initialise the adaptive retriever."""
        self.base_retriever = base_retriever
        self.config = config or AdaptiveConfig()

    # ------------------------------------------------------------------
    # Public API (matches AdvancedRetriever interface)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        return_raw: bool = False,
    ) -> Any:
        """Retrieve with adaptive k-expansion.

        Parameters
        ----------
        query:
            The user query string.
        top_k:
            Override for the final number of results returned.  When
            ``None``, the final k chosen by the adaptive loop is used.
        return_raw:
            If True, return the raw list of chunks instead of wrapping in
            an :class:`~lexisearch.retrieval.advanced.AdvancedRetrievalResult`.

        Returns:
        -------
        AdvancedRetrievalResult (or list[RetrievedChunk] if return_raw=True).
        """
        from lexisearch.retrieval.advanced import AdvancedRetrievalResult

        query_kws = _query_keywords(query)
        history: list[ExpansionStep] = []
        cfg = self.config

        current_k = cfg.initial_k
        chunks: list[Any] = []

        while True:
            chunks = self._fetch(query, current_k)
            coverage = _chunks_coverage(chunks, query_kws)
            spread = _score_spread(chunks)

            step_reason = self._decide(coverage, spread, current_k, cfg)
            history.append(
                ExpansionStep(
                    k=current_k,
                    coverage=coverage,
                    num_chunks=len(chunks),
                    score_spread=spread,
                    reason=step_reason,
                )
            )
            logger.debug(
                "Adaptive k=%d: coverage=%.2f spread=%.4f reason=%s",
                current_k,
                coverage,
                spread,
                step_reason,
            )

            if step_reason != "expand":
                break

            next_k = min(int(current_k * cfg.expand_factor), cfg.max_k)
            if next_k <= current_k:
                # Already at max_k; can't expand further
                break
            current_k = next_k

        final_chunks = chunks[:top_k] if top_k is not None else chunks

        if return_raw:
            return final_chunks

        return AdvancedRetrievalResult(
            query=query,
            chunks=final_chunks,
            sub_queries=[query],
            strategy="adaptive",
            metadata={
                "adaptive": {
                    "final_k": current_k,
                    "final_coverage": history[-1].coverage if history else 0.0,
                    "expansion_steps": len(history),
                    "history": [
                        {
                            "k": s.k,
                            "coverage": s.coverage,
                            "num_chunks": s.num_chunks,
                            "score_spread": s.score_spread,
                            "reason": s.reason,
                        }
                        for s in history
                    ],
                }
            },
        )

    @property
    def expansion_history(self) -> list[ExpansionStep]:
        """Return the expansion history from the last retrieve() call."""
        return self._last_history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _last_history: list[ExpansionStep] = field(default_factory=list)

    def _fetch(self, query: str, top_k: int) -> list[Any]:
        """Call the base retriever, normalising to a plain list."""
        if callable(self.base_retriever):
            raw = self.base_retriever(query, top_k)
        elif hasattr(self.base_retriever, "retrieve"):
            raw = self.base_retriever.retrieve(query, top_k=top_k)
        else:
            raw = self.base_retriever(query, top_k)

        # Unwrap AdvancedRetrievalResult if the base_retriever returned one
        if hasattr(raw, "chunks"):
            return list(raw.chunks)
        return list(raw) if raw else []

    @staticmethod
    def _decide(coverage: float, spread: float, current_k: int, cfg: AdaptiveConfig) -> str:
        """Return 'expand', 'max_k', 'coverage_met', or 'low_spread'."""
        if current_k >= cfg.max_k:
            return "max_k"
        if spread < cfg.score_spread_threshold and current_k > cfg.initial_k:
            # Uniform scores — expanding won't help
            return "low_spread"
        if coverage >= cfg.min_coverage:
            return "coverage_met"
        return "expand"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def adaptive_retriever(
    base_retriever: Any,
    *,
    initial_k: int = 3,
    max_k: int = 20,
    min_coverage: float = 0.6,
    expand_factor: float = 2.0,
) -> AdaptiveRetriever:
    """Create an :class:`AdaptiveRetriever` with common parameters.

    Convenience wrapper for quick one-liner construction.

    Args:
        base_retriever: Callable (query, top_k) or retriever with .retrieve().
        initial_k: Starting k value.
        max_k: Hard upper limit for k.
        min_coverage: Target query keyword coverage before stopping.
        expand_factor: Multiplier applied to k at each expansion step.

    Returns:
        Configured AdaptiveRetriever instance.
    """
    return AdaptiveRetriever(
        base_retriever,
        config=AdaptiveConfig(
            initial_k=initial_k,
            max_k=max_k,
            min_coverage=min_coverage,
            expand_factor=expand_factor,
        ),
    )
