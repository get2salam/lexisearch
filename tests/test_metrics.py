"""Focused tests for the metrics module.

These tests complement the metric tests in test_vectorstore.py with
additional numerical edge cases and property-based checks.
"""

from __future__ import annotations

import math

import pytest

from lexisearch.vectorstore.base import DistanceMetric
from lexisearch.vectorstore.metrics import (
    compute_score,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    l2_normalize,
)


class TestCosineNumericalProperties:
    """Verify numerical properties of cosine similarity."""

    def test_symmetric(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))

    def test_bounded(self) -> None:
        """Cosine similarity should always be in [-1, 1]."""
        vectors = [
            ([1.0, 0.0], [0.0, 1.0]),
            ([1.0, 1.0], [-1.0, -1.0]),
            ([3.0, 4.0], [4.0, 3.0]),
            ([100.0, 0.01], [0.01, 100.0]),
        ]
        for a, b in vectors:
            sim = cosine_similarity(a, b)
            assert -1.0 <= sim <= 1.0, f"Out of bounds: {sim}"

    def test_scale_invariant(self) -> None:
        """Cosine similarity should be invariant to vector magnitude."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        sim1 = cosine_similarity(a, b)
        sim2 = cosine_similarity([x * 10 for x in a], [x * 0.1 for x in b])
        assert sim1 == pytest.approx(sim2, abs=1e-10)


class TestEuclideanNumericalProperties:
    """Verify numerical properties of Euclidean distance."""

    def test_symmetric(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(b, a))

    def test_non_negative(self) -> None:
        a = [1.0, -2.0, 3.0]
        b = [-4.0, 5.0, -6.0]
        assert euclidean_distance(a, b) >= 0.0

    def test_triangle_inequality(self) -> None:
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        c = [1.0, 1.0]
        d_ab = euclidean_distance(a, b)
        d_bc = euclidean_distance(b, c)
        d_ac = euclidean_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-10


class TestDotProductNumericalProperties:
    """Verify numerical properties of dot product."""

    def test_commutative(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert dot_product(a, b) == pytest.approx(dot_product(b, a))

    def test_distributive(self) -> None:
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        c = [5.0, 6.0]
        # a · (b + c) = a · b + a · c
        bc = [b[i] + c[i] for i in range(2)]
        lhs = dot_product(a, bc)
        rhs = dot_product(a, b) + dot_product(a, c)
        assert lhs == pytest.approx(rhs)

    def test_zero_vector(self) -> None:
        assert dot_product([0.0, 0.0], [1.0, 2.0]) == pytest.approx(0.0)


class TestL2NormalizeProperties:
    """Verify properties of L2 normalization."""

    def test_result_has_unit_length(self) -> None:
        vectors = [
            [3.0, 4.0],
            [1.0, 1.0, 1.0],
            [100.0, 0.0],
            [-1.0, -2.0, -3.0, -4.0],
        ]
        for v in vectors:
            normed = l2_normalize(v)
            length = math.sqrt(sum(x * x for x in normed))
            assert length == pytest.approx(1.0)

    def test_preserves_direction(self) -> None:
        v = [3.0, 4.0]
        normed = l2_normalize(v)
        # Angle should be the same — check via cosine similarity
        sim = cosine_similarity(v, normed)
        assert sim == pytest.approx(1.0)

    def test_idempotent(self) -> None:
        v = [3.0, 4.0]
        once = l2_normalize(v)
        twice = l2_normalize(once)
        for a, b in zip(once, twice):
            assert a == pytest.approx(b)


class TestComputeScoreEdgeCases:
    """Edge cases for the score dispatcher."""

    def test_unknown_metric_raises(self) -> None:
        # This requires a non-standard metric which is hard to create
        # with an enum, but we test the error handling
        with pytest.raises((ValueError, KeyError)):
            compute_score([1.0], [1.0], "invalid_metric")  # type: ignore

    def test_euclidean_identical_is_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        score = compute_score(v, v, DistanceMetric.EUCLIDEAN)
        assert score == pytest.approx(1.0)

    def test_euclidean_far_apart_near_zero(self) -> None:
        a = [0.0] * 10
        b = [1000.0] * 10
        score = compute_score(a, b, DistanceMetric.EUCLIDEAN)
        assert score < 0.01
