"""Maximal Marginal Relevance (MMR) for result diversification.

MMR balances relevance to the query with diversity among selected results,
preventing redundant passages from dominating the retrieval output.

References:
    Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based
    reranking for reordering documents and producing summaries. *SIGIR '98*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lexisearch.models import SearchResult
from lexisearch.vectorstore.metrics import cosine_similarity

if TYPE_CHECKING:
    from collections.abc import Sequence


def mmr_select(
    query_vector: Sequence[float],
    candidates: list[SearchResult],
    candidate_vectors: list[Sequence[float]],
    top_k: int = 10,
    lambda_param: float = 0.5,
) -> list[SearchResult]:
    r"""Select results using Maximal Marginal Relevance.

    At each step, MMR selects the candidate that maximises:

    .. math::

        \\text{MMR}(d) = \\lambda \\cdot \\text{sim}(d, q) -
        (1 - \\lambda) \\cdot \\max_{d_j \\in S} \\text{sim}(d, d_j)

    where *q* is the query, *S* is the set of already-selected results,
    and λ controls the relevance-diversity trade-off.

    Args:
        query_vector: The query embedding vector.
        candidates: Search results to select from.
        candidate_vectors: Embedding vectors corresponding to each candidate
            (must be the same length and order as ``candidates``).
        top_k: Number of results to select.
        lambda_param: Trade-off parameter.
            - 1.0 = pure relevance (no diversity).
            - 0.0 = pure diversity (no relevance).
            - 0.5 = balanced (default).

    Returns:
        Selected results in MMR order with updated scores and ranks.

    Raises:
        ValueError: If candidates and vectors have different lengths.
    """
    if len(candidates) != len(candidate_vectors):
        raise ValueError(
            f"Mismatch: {len(candidates)} candidates vs {len(candidate_vectors)} vectors"
        )

    if not candidates:
        return []

    k = min(top_k, len(candidates))

    # Pre-compute query similarities
    query_sims = [cosine_similarity(query_vector, vec) for vec in candidate_vectors]

    selected_indices: list[int] = []
    remaining_indices: set[int] = set(range(len(candidates)))

    for _ in range(k):
        best_idx = -1
        best_mmr = float("-inf")

        for idx in remaining_indices:
            relevance = query_sims[idx]

            # Max similarity to already-selected documents
            if selected_indices:
                max_sim = max(
                    cosine_similarity(
                        candidate_vectors[idx],
                        candidate_vectors[sel_idx],
                    )
                    for sel_idx in selected_indices
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        remaining_indices.discard(best_idx)

    # Build result list
    results: list[SearchResult] = []
    for rank, idx in enumerate(selected_indices, start=1):
        original = candidates[idx]
        results.append(
            SearchResult(
                chunk=original.chunk,
                score=query_sims[idx],
                rank=rank,
                metadata={
                    **original.metadata,
                    "mmr_applied": True,
                    "original_score": original.score,
                    "original_rank": original.rank,
                },
            )
        )

    return results


def greedy_diversify(
    results: list[SearchResult],
    vectors: list[Sequence[float]],
    max_similarity: float = 0.85,
) -> list[SearchResult]:
    """Remove near-duplicate results based on vector similarity.

    Greedily keeps results whose maximum similarity to any already-kept
    result is below the threshold.

    Args:
        results: Ordered search results.
        vectors: Corresponding embedding vectors.
        max_similarity: Similarity threshold — results more similar
            than this to any kept result are discarded.

    Returns:
        Deduplicated results with updated ranks.
    """
    if not results:
        return []

    kept_indices: list[int] = [0]  # Always keep the top result

    for i in range(1, len(results)):
        is_diverse = True
        for j in kept_indices:
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim > max_similarity:
                is_diverse = False
                break

        if is_diverse:
            kept_indices.append(i)

    diverse: list[SearchResult] = []
    for rank, idx in enumerate(kept_indices, start=1):
        original = results[idx]
        diverse.append(
            SearchResult(
                chunk=original.chunk,
                score=original.score,
                rank=rank,
                metadata={
                    **original.metadata,
                    "deduplicated": True,
                    "original_rank": original.rank,
                },
            )
        )

    return diverse
