"""Reranking framework for second-stage result refinement.

Rerankers take an initial set of retrieved results and rescore them
using more expensive but accurate models (e.g., cross-encoders).
This implements the common two-stage retrieval pattern:

1. **Stage 1 (Retriever):** Fast, approximate retrieval (BM25, vector)
2. **Stage 2 (Reranker):** Precise rescoring of top candidates

Supported backends:
    - :class:`CrossEncoderReranker` — Sentence-Transformers cross-encoder.
    - :class:`CohereReranker` — Cohere Rerank API.
    - :class:`LinearScoreReranker` — Simple linear score combination.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from lexisearch.models import SearchResult
from lexisearch.retrieval.base import (
    BaseRetriever,
    MetadataFilter,
    RetrieverConfig,
    RetrieverType,
)


@dataclass
class RerankerConfig:
    """Configuration for reranking.

    Attributes:
        model_name: Identifier for the reranking model.
        top_k: Number of results to return after reranking.
        batch_size: Batch size for cross-encoder inference.
        score_threshold: Minimum reranker score to include a result.
        extra: Backend-specific configuration.
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 10
    batch_size: int = 32
    score_threshold: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """Abstract base class for rerankers.

    Subclasses must implement :meth:`rerank` to rescore a list of
    search results given the original query.
    """

    def __init__(self, config: RerankerConfig | None = None) -> None:
        """Initialize for."""
        self.config = config or RerankerConfig()

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank search results using the underlying model.

        Args:
            query: The original query string.
            results: Initial search results to rerank.
            top_k: Maximum number of results to return.

        Returns:
            Reranked list of :class:`SearchResult` (best first).
        """
        ...

    @abstractmethod
    def score_pair(self, query: str, text: str) -> float:
        """Score a single query-document pair.

        Args:
            query: Query string.
            text: Document text.

        Returns:
            Relevance score.
        """
        ...


class CrossEncoderReranker(BaseReranker):
    """Reranker using Sentence-Transformers cross-encoder models.

    Cross-encoders jointly encode the query and document, producing
    more accurate relevance scores than bi-encoder similarity.

    Args:
        config: Reranker configuration.
        model: Optional pre-loaded cross-encoder model instance. If not
            provided, the model is loaded lazily from ``config.model_name``.

    Note:
        Requires ``sentence-transformers`` to be installed.
    """

    def __init__(
        self,
        config: RerankerConfig | None = None,
        model: Any = None,
    ) -> None:
        """Initialize CrossEncoderReranker."""
        super().__init__(config)
        self._model = model
        self._lazy_loaded = False

    def _ensure_model(self) -> Any:
        """Lazily load the cross-encoder model.

        Returns:
            The loaded cross-encoder model.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if self._model is None and not self._lazy_loaded:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.config.model_name)
                self._lazy_loaded = True
            except ImportError as err:
                raise ImportError(
                    "CrossEncoderReranker requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                ) from err
        return self._model

    def score_pair(self, query: str, text: str) -> float:
        """Score a query-document pair using the cross-encoder.

        Args:
            query: Query string.
            text: Document text.

        Returns:
            Cross-encoder relevance score.
        """
        model = self._ensure_model()
        scores = model.predict([(query, text)])
        return float(scores[0])

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank results using cross-encoder inference.

        Args:
            query: Original query.
            results: Results from stage-1 retrieval.
            top_k: Maximum results to return.

        Returns:
            Reranked results.
        """
        if not results:
            return []

        k = top_k or self.config.top_k
        model = self._ensure_model()

        # Prepare query-document pairs
        pairs = [(query, r.chunk.content) for r in results]

        # Score in batches
        all_scores: list[float] = []
        for i in range(0, len(pairs), self.config.batch_size):
            batch = pairs[i : i + self.config.batch_size]
            scores = model.predict(batch)
            all_scores.extend(float(s) for s in scores)

        # Build scored results
        scored = list(zip(results, all_scores, strict=False))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Apply threshold and limit
        reranked: list[SearchResult] = []
        for rank, (result, score) in enumerate(scored[:k], start=1):
            if score < self.config.score_threshold:
                continue
            reranked.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    rank=rank,
                    metadata={
                        **result.metadata,
                        "reranker": "cross-encoder",
                        "original_score": result.score,
                        "original_rank": result.rank,
                    },
                )
            )

        return reranked


class CohereReranker(BaseReranker):
    """Reranker using the Cohere Rerank API.

    Delegates scoring to Cohere's hosted reranking endpoint.

    Args:
        api_key: Cohere API key.
        config: Reranker configuration.

    Note:
        Requires the ``cohere`` package to be installed.
    """

    def __init__(
        self,
        api_key: str,
        config: RerankerConfig | None = None,
    ) -> None:
        """Initialize CohereReranker."""
        cfg = config or RerankerConfig(model_name="rerank-english-v3.0")
        super().__init__(cfg)
        self._api_key = api_key
        self._client: Any = None

    def _ensure_client(self) -> Any:
        """Lazily initialise the Cohere client.

        Returns:
            Cohere client instance.
        """
        if self._client is None:
            try:
                import cohere

                self._client = cohere.Client(self._api_key)
            except ImportError as err:
                raise ImportError(
                    "CohereReranker requires the cohere package. Install with: pip install cohere"
                ) from err
        return self._client

    def score_pair(self, query: str, text: str) -> float:
        """Score a single query-document pair via Cohere.

        Args:
            query: Query string.
            text: Document text.

        Returns:
            Cohere relevance score.
        """
        client = self._ensure_client()
        response = client.rerank(
            model=self.config.model_name,
            query=query,
            documents=[text],
        )
        return float(response.results[0].relevance_score)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank results using Cohere's reranking API.

        Args:
            query: Original query.
            results: Stage-1 results.
            top_k: Maximum results to return.

        Returns:
            Reranked results.
        """
        if not results:
            return []

        k = top_k or self.config.top_k
        client = self._ensure_client()

        documents = [r.chunk.content for r in results]
        response = client.rerank(
            model=self.config.model_name,
            query=query,
            documents=documents,
            top_n=k,
        )

        reranked: list[SearchResult] = []
        for rank, item in enumerate(response.results, start=1):
            original = results[item.index]
            score = float(item.relevance_score)

            if score < self.config.score_threshold:
                continue

            reranked.append(
                SearchResult(
                    chunk=original.chunk,
                    score=score,
                    rank=rank,
                    metadata={
                        **original.metadata,
                        "reranker": "cohere",
                        "original_score": original.score,
                        "original_rank": original.rank,
                    },
                )
            )

        return reranked


class LinearScoreReranker(BaseReranker):
    """Simple reranker using linear combination of retrieval and text features.

    Combines the original retrieval score with lightweight text-matching
    signals: exact-match bonus, query-term coverage, and length penalty.

    Useful as a baseline or when no ML model is available.

    Args:
        config: Reranker configuration.
        retrieval_weight: Weight for the original retrieval score.
        coverage_weight: Weight for query-term coverage score.
        exact_match_bonus: Bonus for exact query substring match.
        length_penalty: Penalty factor for very long documents.
    """

    def __init__(
        self,
        config: RerankerConfig | None = None,
        retrieval_weight: float = 0.6,
        coverage_weight: float = 0.3,
        exact_match_bonus: float = 0.1,
        length_penalty: float = 0.001,
    ) -> None:
        """Initialize LinearScoreReranker."""
        super().__init__(config)
        self.retrieval_weight = retrieval_weight
        self.coverage_weight = coverage_weight
        self.exact_match_bonus = exact_match_bonus
        self.length_penalty = length_penalty

    def score_pair(self, query: str, text: str) -> float:
        """Score a query-document pair using text features.

        Args:
            query: Query string.
            text: Document text.

        Returns:
            Combined feature score.
        """
        query_lower = query.lower()
        text_lower = text.lower()

        # Query term coverage
        query_terms = set(query_lower.split())
        if query_terms:
            covered = sum(1 for t in query_terms if t in text_lower)
            coverage = covered / len(query_terms)
        else:
            coverage = 0.0

        # Exact match
        exact = 1.0 if query_lower in text_lower else 0.0

        # Length penalty (prefer concise, relevant passages)
        word_count = len(text.split())
        penalty = max(0.0, 1.0 - self.length_penalty * max(0, word_count - 200))

        return self.coverage_weight * coverage + self.exact_match_bonus * exact + penalty * 0.05

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank using linear feature combination.

        Args:
            query: Original query.
            results: Stage-1 results.
            top_k: Maximum results.

        Returns:
            Reranked results.
        """
        if not results:
            return []

        k = top_k or self.config.top_k

        scored: list[tuple[SearchResult, float]] = []
        for result in results:
            feature_score = self.score_pair(query, result.chunk.content)
            combined = self.retrieval_weight * result.score + feature_score
            scored.append((result, combined))

        scored.sort(key=lambda x: x[1], reverse=True)

        reranked: list[SearchResult] = []
        for rank, (result, score) in enumerate(scored[:k], start=1):
            reranked.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    rank=rank,
                    metadata={
                        **result.metadata,
                        "reranker": "linear",
                        "original_score": result.score,
                        "original_rank": result.rank,
                    },
                )
            )

        return reranked


class RerankedRetriever(BaseRetriever):
    """Retriever that wraps a base retriever with a reranking stage.

    Implements the two-stage retrieval pattern: the base retriever
    fetches candidates, then the reranker rescores the top results.

    Args:
        retriever: The stage-1 retriever.
        reranker: The stage-2 reranker.
        prefetch_multiplier: How many more results to fetch from stage-1
            relative to the final ``top_k``.
        config: Retriever configuration.

    Example::

        base = VectorRetriever(store, embedder)
        reranker = LinearScoreReranker()
        pipeline = RerankedRetriever(base, reranker, prefetch_multiplier=3)
        results = pipeline.retrieve("neural network architectures")
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        prefetch_multiplier: int = 3,
        config: RetrieverConfig | None = None,
    ) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.retriever = retriever
        self.reranker = reranker
        self.prefetch_multiplier = prefetch_multiplier

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: list[MetadataFilter] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Retrieve and rerank results.

        Args:
            query: Natural-language query.
            top_k: Final number of results desired.
            filters: Metadata filters for stage-1.
            **kwargs: Passed to the base retriever.

        Returns:
            Reranked results.
        """
        k = top_k or self.config.top_k
        prefetch_k = k * self.prefetch_multiplier

        # Stage 1: retrieve candidates
        candidates = self.retriever.retrieve(query, top_k=prefetch_k, filters=filters, **kwargs)

        # Stage 2: rerank
        return self.reranker.rerank(query, candidates, top_k=k)

    def retriever_type(self) -> RetrieverType:
        """Return the retriever type.

        Returns:
            :attr:`RetrieverType.RERANKED`
        """
        return RetrieverType.RERANKED

    def get_config(self) -> dict[str, Any]:
        """Return the reranked retriever configuration."""
        base = super().get_config()
        base.update(
            {
                "base_retriever": self.retriever.retriever_type().value,
                "reranker": type(self.reranker).__name__,
                "prefetch_multiplier": self.prefetch_multiplier,
            }
        )
        return base
