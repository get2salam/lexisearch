"""Hybrid retriever combining sparse and dense search.

Fuses results from multiple retrievers using configurable strategies:
reciprocal rank fusion (RRF), linear weighted combination, or
distribution-based score fusion (DBSF).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lexisearch.models import Chunk, SearchResult
from lexisearch.retrieval.base import (
    BaseRetriever,
    MetadataFilter,
    RetrieverConfig,
    RetrieverType,
)


class FusionMethod(Enum):
    """Strategies for combining scores from multiple retrievers."""

    RRF = "rrf"
    """Reciprocal Rank Fusion — rank-based, parameter-free beyond *k*."""

    LINEAR = "linear"
    """Weighted linear combination of normalised scores."""

    DBSF = "dbsf"
    """Distribution-Based Score Fusion — normalise per-retriever
    score distributions before combining."""


@dataclass
class HybridConfig(RetrieverConfig):
    """Configuration for the hybrid retriever.

    Attributes:
        fusion_method: Strategy for combining retriever scores.
        weights: Per-retriever weights for LINEAR / DBSF fusion.
            Must have the same length as the list of retrievers.
            Defaults to equal weighting.
        rrf_k: The *k* constant for RRF (higher = more uniform blending).
            Standard default is 60.
        pre_fetch_multiplier: Multiplier on ``top_k`` for pre-fetching
            from each sub-retriever to ensure enough candidates.
    """

    fusion_method: FusionMethod = FusionMethod.RRF
    weights: list[float] = field(default_factory=list)
    rrf_k: int = 60
    pre_fetch_multiplier: int = 3


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining multiple sub-retrievers via score fusion.

    Given *N* retrievers, each independently scores and ranks candidates.
    The hybrid retriever fuses the results into a single ranked list.

    Args:
        retrievers: List of sub-retrievers to combine.
        config: Hybrid-specific configuration.

    Example::

        hybrid = HybridRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            config=HybridConfig(
                fusion_method=FusionMethod.RRF,
                top_k=10,
            ),
        )
        results = hybrid.retrieve("machine learning applications")
    """

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        config: HybridConfig | None = None,
    ) -> None:
        """Initialize the instance."""
        cfg = config or HybridConfig()
        super().__init__(cfg)
        self._config = cfg

        if len(retrievers) < 2:
            raise ValueError(
                f"HybridRetriever requires at least 2 sub-retrievers, got {len(retrievers)}"
            )

        self.retrievers = retrievers

        # Default to equal weights if not specified
        if not cfg.weights:
            cfg.weights = [1.0 / len(retrievers)] * len(retrievers)
        elif len(cfg.weights) != len(retrievers):
            raise ValueError(f"Expected {len(retrievers)} weights, got {len(cfg.weights)}")

    # ------------------------------------------------------------------
    # Fusion strategies
    # ------------------------------------------------------------------

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: list[list[SearchResult]],
        k: int = 60,
    ) -> list[tuple[str, float, Chunk]]:
        r"""Combine results using Reciprocal Rank Fusion.

        .. math::
            \\text{RRF}(d) = \\sum_{r \\in R} \\frac{1}{k + \\text{rank}_r(d)}

        Args:
            result_lists: List of ranked result lists from each retriever.
            k: The smoothing constant (default 60).

        Returns:
            Fused results as ``(chunk_id, rrf_score, chunk)`` tuples,
            sorted by score descending.
        """
        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                cid = result.chunk.id
                chunk_map[cid] = result.chunk
                rrf_score = 1.0 / (k + rank)
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + rrf_score

        fused = [(cid, score, chunk_map[cid]) for cid, score in chunk_scores.items()]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    @staticmethod
    def linear_fusion(
        result_lists: list[list[SearchResult]],
        weights: list[float],
    ) -> list[tuple[str, float, Chunk]]:
        """Combine results using weighted linear score fusion.

        Each retriever's scores are min-max normalised to [0, 1] before
        applying weights.

        Args:
            result_lists: Ranked result lists from each retriever.
            weights: Per-retriever weights (must sum to 1 ideally).

        Returns:
            Fused results sorted by combined score descending.
        """
        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for weight, results in zip(weights, result_lists, strict=False):
            if not results:
                continue

            # Min-max normalise scores within this retriever
            scores = [r.score for r in results]
            min_s = min(scores)
            max_s = max(scores)
            rng = max_s - min_s if max_s != min_s else 1.0

            for result in results:
                cid = result.chunk.id
                chunk_map[cid] = result.chunk
                norm_score = (result.score - min_s) / rng
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + weight * norm_score

        fused = [(cid, score, chunk_map[cid]) for cid, score in chunk_scores.items()]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    @staticmethod
    def dbsf_fusion(
        result_lists: list[list[SearchResult]],
        weights: list[float],
    ) -> list[tuple[str, float, Chunk]]:
        """Combine results using Distribution-Based Score Fusion.

        Normalises each retriever's scores using z-score normalisation
        (mean=0, std=1) then applies weights. More robust than min-max
        when score distributions differ significantly.

        Args:
            result_lists: Ranked result lists from each retriever.
            weights: Per-retriever weights.

        Returns:
            Fused results sorted by combined score descending.
        """
        import math

        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for weight, results in zip(weights, result_lists, strict=False):
            if not results:
                continue

            scores = [r.score for r in results]
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            std = math.sqrt(variance) if variance > 0 else 1.0

            for result in results:
                cid = result.chunk.id
                chunk_map[cid] = result.chunk
                z_score = (result.score - mean) / std
                # Shift to positive range: sigmoid-like transform
                norm_score = 1.0 / (1.0 + math.exp(-z_score))
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + weight * norm_score

        fused = [(cid, score, chunk_map[cid]) for cid, score in chunk_scores.items()]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    # ------------------------------------------------------------------
    # Retriever interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: list[MetadataFilter] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Retrieve chunks using hybrid fusion of multiple retrievers.

        Each sub-retriever runs independently, then results are fused
        using the configured fusion method.

        Args:
            query: Natural-language query.
            top_k: Maximum number of final results.
            filters: Metadata filters passed to all sub-retrievers.
            **kwargs: Additional parameters for sub-retrievers.

        Returns:
            Fused and ranked list of :class:`SearchResult`.
        """
        k = top_k or self.config.top_k
        prefetch_k = k * self._config.pre_fetch_multiplier

        # Collect results from all sub-retrievers
        all_results: list[list[SearchResult]] = []
        retriever_names: list[str] = []

        for retriever in self.retrievers:
            results = retriever.retrieve(query, top_k=prefetch_k, filters=filters, **kwargs)
            all_results.append(results)
            retriever_names.append(retriever.retriever_type().value)

        # Fuse results
        method = self._config.fusion_method
        if method is FusionMethod.RRF:
            fused = self.reciprocal_rank_fusion(all_results, self._config.rrf_k)
        elif method is FusionMethod.LINEAR:
            fused = self.linear_fusion(all_results, self._config.weights)
        elif method is FusionMethod.DBSF:
            fused = self.dbsf_fusion(all_results, self._config.weights)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        # Build SearchResult objects
        results: list[SearchResult] = []
        for rank, (chunk_id, score, chunk) in enumerate(fused[:k], start=1):
            # Track which retrievers contributed to this result
            sources: list[str] = []
            for name, result_list in zip(retriever_names, all_results, strict=False):
                if any(r.chunk.id == chunk_id for r in result_list):
                    sources.append(name)

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    metadata={
                        "retriever": "hybrid",
                        "fusion_method": method.value,
                        "sources": sources,
                    },
                )
            )

        return results

    def retriever_type(self) -> RetrieverType:
        """Return the retriever type.

        Returns:
            :attr:`RetrieverType.HYBRID`
        """
        return RetrieverType.HYBRID

    def get_config(self) -> dict[str, Any]:
        """Return the hybrid retriever configuration.

        Returns:
            Configuration dictionary including fusion method and sub-retrievers.
        """
        base = super().get_config()
        base.update(
            {
                "fusion_method": self._config.fusion_method.value,
                "weights": self._config.weights,
                "rrf_k": self._config.rrf_k,
                "num_retrievers": len(self.retrievers),
                "sub_retrievers": [r.retriever_type().value for r in self.retrievers],
            }
        )
        return base
