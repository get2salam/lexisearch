"""Similarity and distance metric functions.

Pure-Python implementations used by the in-memory vector store and as
reference implementations for testing third-party backends.

All functions operate on plain ``list[float]`` vectors so there are **no**
NumPy dependencies required at this layer.
"""

from __future__ import annotations

import math
from typing import Sequence

from lexisearch.vectorstore.base import DistanceMetric


# ------------------------------------------------------------------
# Core metric functions
# ------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors.

    .. math::
        \\text{sim}(a, b) = \\frac{a \\cdot b}{\\|a\\| \\, \\|b\\|}

    Args:
        a: First vector.
        b: Second vector (must have the same length as *a*).

    Returns:
        Similarity in the range ``[-1, 1]``.

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector length mismatch: {len(a)} vs {len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Euclidean (L2) distance between two vectors.

    .. math::
        d(a, b) = \\sqrt{\\sum_i (a_i - b_i)^2}

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Non-negative distance (0 means identical).

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector length mismatch: {len(a)} vs {len(b)}"
        )
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the inner (dot) product of two vectors.

    .. math::
        \\text{dot}(a, b) = \\sum_i a_i \\, b_i

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The dot product value.

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector length mismatch: {len(a)} vs {len(b)}"
        )
    return sum(x * y for x, y in zip(a, b))


# ------------------------------------------------------------------
# Normalisation helpers
# ------------------------------------------------------------------


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Return an L2-normalised copy of *vector*.

    Args:
        vector: Input vector.

    Returns:
        Unit-length vector (or zero vector if input is zero).
    """
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return [0.0] * len(vector)
    return [x / norm for x in vector]


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------


def compute_score(
    a: Sequence[float],
    b: Sequence[float],
    metric: DistanceMetric,
) -> float:
    """Compute a similarity score using the given metric.

    For :attr:`DistanceMetric.EUCLIDEAN` the score is converted so that
    *higher means more similar*:

    .. math::
        \\text{score} = \\frac{1}{1 + d}

    Args:
        a: First vector.
        b: Second vector.
        metric: The distance metric to use.

    Returns:
        Similarity score where **higher is better**.
    """
    if metric is DistanceMetric.COSINE:
        return cosine_similarity(a, b)
    if metric is DistanceMetric.DOT_PRODUCT:
        return dot_product(a, b)
    if metric is DistanceMetric.EUCLIDEAN:
        dist = euclidean_distance(a, b)
        return 1.0 / (1.0 + dist)
    raise ValueError(f"Unknown metric: {metric}")


def compute_pairwise_scores(
    query: Sequence[float],
    vectors: list[Sequence[float]],
    metric: DistanceMetric,
) -> list[float]:
    """Compute similarity scores between a query and a list of vectors.

    Args:
        query: Query vector.
        vectors: Candidate vectors.
        metric: Distance metric.

    Returns:
        List of similarity scores (one per candidate).
    """
    return [compute_score(query, v, metric) for v in vectors]
