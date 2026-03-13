"""Standard information-retrieval metrics for LexiSearch.

Implements Recall@k, Precision@k, MRR, AP, MAP, and NDCG@k.
These measure *retrieval quality*, complementing the generation metrics
in :mod:`lexisearch.evaluation.metrics` which measure *answer quality*.

All per-query functions share a consistent signature::

    metric(retrieved_ids, relevant_ids, *, k=None) -> float

where ``retrieved_ids`` is an ordered list of IDs (best first) and
``relevant_ids`` is a set of ground-truth relevant IDs.

Example::

    from lexisearch.evaluation.ir_metrics import (
        recall_at_k, mrr, ndcg_at_k, compute_ir_metrics,
    )

    retrieved = ["doc3", "doc1", "doc5", "doc2", "doc4"]
    relevant  = {"doc1", "doc3"}

    print(recall_at_k(retrieved, relevant, k=5))   # 1.0
    print(mrr([retrieved], [relevant]))             # 1.0
    print(ndcg_at_k(retrieved, relevant, k=5))      # ~0.93
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------


def recall_at_k(
    retrieved: Sequence[str],
    relevant: set[str] | frozenset[str] | Sequence[str],
    *,
    k: int | None = None,
) -> float:
    """Fraction of relevant documents found in the top-k results.

    Args:
        retrieved: Ordered retrieved IDs, best first.
        relevant: Ground-truth relevant IDs.
        k: Rank cutoff. Defaults to len(retrieved).

    Returns:
        Score in [0.0, 1.0]. Returns 0.0 when relevant is empty.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(retrieved)[:k] if k is not None else list(retrieved)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(
    retrieved: Sequence[str],
    relevant: set[str] | frozenset[str] | Sequence[str],
    *,
    k: int | None = None,
) -> float:
    """Fraction of top-k retrieved results that are relevant.

    Args:
        retrieved: Ordered retrieved IDs, best first.
        relevant: Ground-truth relevant IDs.
        k: Rank cutoff. Defaults to len(retrieved).

    Returns:
        Score in [0.0, 1.0]. Returns 0.0 when retrieved is empty.
    """
    relevant_set = set(relevant)
    top_k = list(retrieved)[:k] if k is not None else list(retrieved)
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(top_k)


def average_precision(
    retrieved: Sequence[str],
    relevant: set[str] | frozenset[str] | Sequence[str],
    *,
    k: int | None = None,
) -> float:
    """Average precision (AP) for a single query.

    Computes the mean of precision values at each rank position where
    a relevant document appears (area under the precision-recall curve).

    Args:
        retrieved: Ordered retrieved IDs, best first.
        relevant: Ground-truth relevant IDs.
        k: Rank cutoff. Defaults to len(retrieved).

    Returns:
        AP in [0.0, 1.0]. Returns 0.0 when relevant is empty.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(retrieved)[:k] if k is not None else list(retrieved)
    cumulative_precision = 0.0
    hits = 0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_set:
            hits += 1
            cumulative_precision += hits / rank
    if hits == 0:
        return 0.0
    return cumulative_precision / len(relevant_set)


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: set[str] | frozenset[str] | Sequence[str] | dict[str, float],
    *,
    k: int | None = None,
) -> float:
    """Normalised Discounted Cumulative Gain at k (NDCG@k).

    Supports binary relevance (set/list) or graded relevance (dict
    mapping ID to a float score, e.g. 0-3).  Higher-ranked relevant
    documents contribute more than lower-ranked ones.

    Args:
        retrieved: Ordered retrieved IDs, best first.
        relevant: Binary (set/list) or graded (dict[str, float]) relevance.
        k: Rank cutoff. Defaults to len(retrieved).

    Returns:
        NDCG in [0.0, 1.0].
    """
    if isinstance(relevant, dict):
        rel_scores: dict[str, float] = relevant
    else:
        rel_scores = {doc_id: 1.0 for doc_id in relevant}
    if not rel_scores:
        return 0.0
    top_k = list(retrieved)[:k] if k is not None else list(retrieved)
    cutoff = len(top_k)

    def _dcg(order: list[str]) -> float:
        return sum(
            rel_scores.get(doc_id, 0.0) / math.log2(rank + 1)
            for rank, doc_id in enumerate(order, start=1)
        )

    dcg = _dcg(top_k)
    ideal_order = sorted(rel_scores, key=lambda x: rel_scores[x], reverse=True)[:cutoff]
    ideal_dcg = _dcg(ideal_order)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: set[str] | frozenset[str] | Sequence[str],
) -> float:
    """Reciprocal rank (1/rank of first relevant) for a single query.

    Args:
        retrieved: Ordered retrieved IDs, best first.
        relevant: Ground-truth relevant IDs.

    Returns:
        1/rank of first relevant result, or 0.0 if none found.
    """
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Batch / aggregate metrics
# ---------------------------------------------------------------------------


def mrr(
    all_retrieved: Sequence[Sequence[str]],
    all_relevant: Sequence[set[str] | frozenset[str] | Sequence[str]],
) -> float:
    """Mean Reciprocal Rank (MRR) across a set of queries.

    Args:
        all_retrieved: One ordered retrieved list per query.
        all_relevant: Corresponding relevant-ID sets, one per query.

    Returns:
        MRR in [0.0, 1.0]. Returns 0.0 on empty input.
    """
    if not all_retrieved:
        return 0.0
    rr_sum = sum(
        reciprocal_rank(retr, rel)
        for retr, rel in zip(all_retrieved, all_relevant)
    )
    return rr_sum / len(all_retrieved)


def mean_average_precision(
    all_retrieved: Sequence[Sequence[str]],
    all_relevant: Sequence[set[str] | frozenset[str] | Sequence[str]],
    *,
    k: int | None = None,
) -> float:
    """Mean Average Precision (MAP) across a set of queries.

    Args:
        all_retrieved: One ordered retrieved list per query.
        all_relevant: Corresponding relevant-ID sets, one per query.
        k: Optional rank cutoff applied to each AP computation.

    Returns:
        MAP in [0.0, 1.0]. Returns 0.0 on empty input.
    """
    if not all_retrieved:
        return 0.0
    ap_sum = sum(
        average_precision(retr, rel, k=k)
        for retr, rel in zip(all_retrieved, all_relevant)
    )
    return ap_sum / len(all_retrieved)


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


@dataclass
class RetrievalEvalReport:
    """Aggregated IR evaluation results across multiple queries."""

    num_queries: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    map_score: float
    ndcg_at_k: float
    k: int
    per_query: list[dict[str, float]] = field(default_factory=list)
    """Per-query breakdowns (populated when include_per_query=True)."""

    def summary(self) -> str:
        """Return a human-readable one-line summary of all metrics."""
        return (
            f"Queries={self.num_queries} k={self.k} | "
            f"R@k={self.recall_at_k:.3f} P@k={self.precision_at_k:.3f} "
            f"MRR={self.mrr:.3f} MAP={self.map_score:.3f} "
            f"NDCG@k={self.ndcg_at_k:.3f}"
        )


def compute_ir_metrics(
    all_retrieved: Sequence[Sequence[str]],
    all_relevant: Sequence[set[str] | frozenset[str] | Sequence[str]],
    *,
    k: int = 10,
    include_per_query: bool = False,
) -> RetrievalEvalReport:
    """Compute a full suite of IR metrics for a batch of queries.

    Args:
        all_retrieved: Ordered retrieved IDs per query (best first).
        all_relevant: Corresponding ground-truth relevant ID sets.
        k: Rank cutoff applied to all metrics.
        include_per_query: If True, populate RetrievalEvalReport.per_query.

    Returns:
        RetrievalEvalReport with aggregate and optional per-query results.

    Raises:
        ValueError: If all_retrieved and all_relevant differ in length.
    """
    if len(all_retrieved) != len(all_relevant):
        msg = (
            f"all_retrieved ({len(all_retrieved)}) and "
            f"all_relevant ({len(all_relevant)}) must have the same length"
        )
        raise ValueError(msg)

    n = len(all_retrieved)
    if n == 0:
        return RetrievalEvalReport(
            num_queries=0,
            recall_at_k=0.0,
            precision_at_k=0.0,
            mrr=0.0,
            map_score=0.0,
            ndcg_at_k=0.0,
            k=k,
        )

    per_query: list[dict[str, float]] = []
    recall_sum = prec_sum = ndcg_sum = 0.0

    for retrieved, relevant in zip(all_retrieved, all_relevant):
        r = recall_at_k(retrieved, relevant, k=k)
        p = precision_at_k(retrieved, relevant, k=k)
        nd = ndcg_at_k(retrieved, relevant, k=k)
        recall_sum += r
        prec_sum += p
        ndcg_sum += nd
        if include_per_query:
            per_query.append(
                {
                    "recall_at_k": r,
                    "precision_at_k": p,
                    "ap": average_precision(retrieved, relevant, k=k),
                    "rr": reciprocal_rank(retrieved, relevant),
                    "ndcg_at_k": nd,
                }
            )

    return RetrievalEvalReport(
        num_queries=n,
        recall_at_k=recall_sum / n,
        precision_at_k=prec_sum / n,
        mrr=mrr(all_retrieved, all_relevant),
        map_score=mean_average_precision(all_retrieved, all_relevant, k=k),
        ndcg_at_k=ndcg_sum / n,
        k=k,
        per_query=per_query,
    )
