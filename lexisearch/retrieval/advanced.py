"""Advanced retrieval strategies for LexiSearch.

Implements three higher-order retrieval techniques that improve recall and
relevance beyond vanilla dense search:

HyDE (Hypothetical Document Embeddings)
    Rather than embedding the query directly, a *hypothetical* answer
    document is generated and then embedded.  The idea is that the
    hypothetical document sits closer to real answer chunks in embedding
    space than a short question does.

    Reference: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without
    Relevance Labels" <https://arxiv.org/abs/2212.10496>

Step-Back Prompting
    The raw query is transformed into a broader / more abstract form ("step
    back") before retrieval.  Results from both the original and step-back
    queries are fused, boosting recall for queries that are too specific for
    an exact match but well-covered by higher-level chunks.

    Reference: Zheng et al. (2023) "Take a Step Back: Evoking Reasoning via
    Abstraction in Large Language Models"

Multi-Query Retrieval
    The query is paraphrased into ``n`` alternative formulations.  Each
    variant is retrieved independently and results are fused via reciprocal
    rank fusion (RRF).  This reduces sensitivity to phrasing and consistently
    improves recall@k.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------


@dataclass
class AdvancedRetrievalConfig:
    """Configuration shared by all advanced retrieval strategies."""

    top_k: int = 5
    """Number of final results to return."""

    overretrieve_factor: int = 3
    """Multiplier used when fetching candidates before fusion/reranking.
    E.g., ``top_k=5, overretrieve_factor=3`` → fetch 15 candidates per sub-query."""

    rrf_k: int = 60
    """Reciprocal rank fusion constant (higher = gentler rank penalty)."""

    deduplicate: bool = True
    """Remove duplicate chunks (by content hash) before returning results."""


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its score and provenance."""

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source_query: str = ""
    """The query variation that retrieved this chunk (for debugging)."""


@dataclass
class AdvancedRetrievalResult:
    """Aggregated result from an advanced retrieval pass."""

    query: str
    chunks: list[RetrievedChunk]
    sub_queries: list[str] = field(default_factory=list)
    strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _content_hash(text: str) -> str:
    """Return a short content fingerprint for deduplication."""
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:12]


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    k: int = 60,
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists via reciprocal rank fusion.

    Parameters
    ----------
    ranked_lists:
        Each element is an ordered list of ``RetrievedChunk`` (best first).
    k:
        RRF constant (typically 60).
    top_n:
        Return only the top ``top_n`` results.  ``None`` returns all.

    Returns:
    -------
    list[RetrievedChunk]
        Merged and re-ranked chunks, best first.
    """
    scores: dict[str, float] = defaultdict(float)
    representatives: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk.chunk_id or _content_hash(chunk.content)
            scores[cid] += 1.0 / (k + rank)
            if cid not in representatives:
                representatives[cid] = chunk

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    if top_n is not None:
        sorted_ids = sorted_ids[:top_n]

    result: list[RetrievedChunk] = []
    for cid in sorted_ids:
        chunk = representatives[cid]
        result.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=scores[cid],
                metadata=chunk.metadata,
                source_query=chunk.source_query,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseAdvancedRetriever:
    """Abstract base for advanced retrieval strategies.

    Subclasses implement ``_fetch`` (the low-level retrieval call that returns
    ``RetrievedChunk`` lists) and ``retrieve`` (the high-level strategy).
    """

    def __init__(self, config: AdvancedRetrievalConfig | None = None) -> None:
        """Initialise with optional configuration."""
        self.config = config or AdvancedRetrievalConfig()

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int | None = None) -> AdvancedRetrievalResult:
        """Run the advanced retrieval strategy and return results."""

    def _dedup(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove exact-content duplicates, preserving order."""
        if not self.config.deduplicate:
            return chunks
        seen: set[str] = set()
        out: list[RetrievedChunk] = []
        for c in chunks:
            h = _content_hash(c.content)
            if h not in seen:
                seen.add(h)
                out.append(c)
        return out


# ---------------------------------------------------------------------------
# Query generator (rule-based fallback, no LLM required)
# ---------------------------------------------------------------------------


class RuleBasedQueryGenerator:
    """Generate query variants without an LLM using heuristic rules.

    Useful as a zero-dependency fallback or for testing.
    """

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
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "and",
            "or",
            "but",
            "if",
            "that",
            "this",
            "these",
            "those",
            "not",
        }
    )

    def paraphrase(self, query: str, n: int = 3) -> list[str]:
        """Return up to ``n`` paraphrases of *query* using rule-based transforms."""
        variants: list[str] = []

        # Variant 1: keyword extraction (remove stop words + question words)
        tokens = re.findall(r"\w+", query.lower())
        keywords = [t for t in tokens if t not in self._STOP_WORDS and len(t) > 2]
        if keywords and keywords != tokens:
            variants.append(" ".join(keywords))

        # Variant 2: rephrase question as declarative
        q = query.strip().rstrip("?")
        for prefix in ("what is", "what are", "how does", "how do", "explain", "describe"):
            if q.lower().startswith(prefix):
                rest = q[len(prefix) :].strip()
                variants.append(f"information about {rest}")
                break

        # Variant 3: add "definition of" prefix for short queries
        words = query.split()
        if len(words) <= 4:
            variants.append(f"definition of {query.lower().rstrip('?')}")

        return [v for v in variants if v.strip() and v.lower() != query.lower()][:n]

    def step_back(self, query: str) -> str:
        """Return a broader / more abstract version of *query*."""
        q = query.strip().rstrip("?")

        # Drop specifics after "in", "for", "with" etc.
        for connector in (" in ", " for ", " using ", " with ", " of ", " about "):
            if connector in q.lower():
                idx = q.lower().index(connector)
                broader = q[:idx].strip()
                if broader:
                    return broader

        # Try removing adjective-like modifiers (simple heuristic)
        tokens = q.split()
        if len(tokens) > 3:
            # Drop first token if it looks like a modifier (short, not a noun starter)
            return " ".join(tokens[1:])

        return q

    def generate_hypothesis(self, query: str) -> str:
        """Return a plausible hypothetical answer sentence for HyDE."""
        q = query.strip().rstrip("?")
        # Very simple: create a declarative sentence
        q_lower = q.lower()
        for prefix in ("what is", "what are"):
            if q_lower.startswith(prefix):
                rest = q[len(prefix) :].strip()
                return f"{rest.capitalize()} is a concept related to {rest.lower()}."
        for prefix in ("how does", "how do"):
            if q_lower.startswith(prefix):
                rest = q[len(prefix) :].strip()
                return f"{rest.capitalize()} works by performing the relevant operations."
        # Fallback
        return f"The answer to '{q}' involves relevant information about the topic."


# ---------------------------------------------------------------------------
# HyDE retriever
# ---------------------------------------------------------------------------


class HyDERetriever(BaseAdvancedRetriever):
    """Hypothetical Document Embedding retriever.

    Instead of embedding the query text directly, we first generate a
    *hypothetical answer* and embed that.  The hypothetical answer lives
    closer to real answer chunks in embedding space than a terse question
    does — leading to higher recall.

    Parameters
    ----------
    base_retriever:
        Any callable ``(query: str, top_k: int) -> list[RetrievedChunk]``.
        Typically wraps a ``VectorRetriever`` or ``HybridRetriever``.
    embedder:
        Optional embedding function ``(text: str) -> list[float]``.  When
        supplied, the hypothetical document is embedded and the closest
        chunks are retrieved via the embedder.  When ``None``, the
        hypothetical text is passed directly to ``base_retriever``.
    generator:
        Object with a ``generate_hypothesis(query) -> str`` method.
        Defaults to ``RuleBasedQueryGenerator``.
    config:
        Shared configuration.
    """

    def __init__(
        self,
        base_retriever: Any,
        embedder: Any | None = None,
        generator: Any | None = None,
        config: AdvancedRetrievalConfig | None = None,
    ) -> None:
        """Initialise HyDE retriever with a base retriever and optional components."""
        super().__init__(config)
        self.base_retriever = base_retriever
        self.embedder = embedder
        self.generator = generator or RuleBasedQueryGenerator()

    def retrieve(self, query: str, *, top_k: int | None = None) -> AdvancedRetrievalResult:
        """Retrieve using a hypothetical document embedding."""
        k = top_k or self.config.top_k
        candidate_k = k * self.config.overretrieve_factor

        # Step 1: generate hypothesis
        hypothesis = self.generator.generate_hypothesis(query)
        logger.debug("HyDE hypothesis: %r", hypothesis[:120])

        # Step 2: retrieve using hypothesis text (or embedding)
        try:
            hyde_chunks = self._fetch(hypothesis, candidate_k)
        except Exception:
            logger.warning("HyDE hypothesis retrieval failed, falling back to direct query")
            hyde_chunks = self._fetch(query, candidate_k)

        # Step 3: also retrieve with the original query for safety
        original_chunks = self._fetch(query, candidate_k)

        # Step 4: fuse via RRF
        fused = reciprocal_rank_fusion(
            [hyde_chunks, original_chunks],
            k=self.config.rrf_k,
            top_n=k,
        )
        deduped = self._dedup(fused)

        return AdvancedRetrievalResult(
            query=query,
            chunks=deduped[:k],
            sub_queries=[hypothesis, query],
            strategy="hyde",
            metadata={"hypothesis": hypothesis},
        )

    def _fetch(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Delegate to the base retriever."""
        raw = self.base_retriever(query, top_k)
        return self._normalise(raw, source_query=query)

    @staticmethod
    def _normalise(raw: Any, *, source_query: str = "") -> list[RetrievedChunk]:
        """Convert various retriever output formats to ``RetrievedChunk``."""
        results: list[RetrievedChunk] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, RetrievedChunk):
                    results.append(item)
                elif hasattr(item, "content"):
                    results.append(
                        RetrievedChunk(
                            chunk_id=getattr(item, "chunk_id", "") or _content_hash(item.content),
                            content=str(item.content),
                            score=float(getattr(item, "score", 0.0)),
                            metadata=dict(getattr(item, "metadata", {})),
                            source_query=source_query,
                        )
                    )
        return results


# ---------------------------------------------------------------------------
# Step-Back retriever
# ---------------------------------------------------------------------------


class StepBackRetriever(BaseAdvancedRetriever):
    """Step-Back prompting retriever.

    Performs two retrieval passes:
    1. Original query → focused results
    2. Step-back (broader) query → contextual background results

    Results are fused via RRF, giving the final answer both specific facts
    and relevant background context.

    Parameters
    ----------
    base_retriever:
        Callable ``(query: str, top_k: int) -> list[RetrievedChunk]``.
    generator:
        Object with a ``step_back(query) -> str`` method.
    config:
        Shared configuration.
    """

    def __init__(
        self,
        base_retriever: Any,
        generator: Any | None = None,
        config: AdvancedRetrievalConfig | None = None,
    ) -> None:
        """Initialise Step-Back retriever with a base retriever and optional generator."""
        super().__init__(config)
        self.base_retriever = base_retriever
        self.generator = generator or RuleBasedQueryGenerator()

    def retrieve(self, query: str, *, top_k: int | None = None) -> AdvancedRetrievalResult:
        """Retrieve using both the original and step-back query."""
        k = top_k or self.config.top_k
        candidate_k = k * self.config.overretrieve_factor

        # Generate the step-back (broader) form
        step_back_query = self.generator.step_back(query)
        logger.debug("Step-back query: %r → %r", query, step_back_query)

        # Retrieve with both queries
        original_chunks = self._fetch(query, candidate_k)
        step_back_chunks = (
            self._fetch(step_back_query, candidate_k)
            if step_back_query.lower() != query.lower()
            else []
        )

        # Fuse
        fused = reciprocal_rank_fusion(
            [original_chunks, step_back_chunks],
            k=self.config.rrf_k,
            top_n=k,
        )
        deduped = self._dedup(fused)

        return AdvancedRetrievalResult(
            query=query,
            chunks=deduped[:k],
            sub_queries=[query, step_back_query],
            strategy="step_back",
            metadata={"step_back_query": step_back_query},
        )

    def _fetch(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raw = self.base_retriever(query, top_k)
        return HyDERetriever._normalise(raw, source_query=query)


# ---------------------------------------------------------------------------
# Multi-Query retriever
# ---------------------------------------------------------------------------


class MultiQueryRetriever(BaseAdvancedRetriever):
    """Multi-query retrieval with reciprocal rank fusion.

    Generates ``num_variants`` alternative phrasings of the query, retrieves
    for each, and fuses results via RRF.  This is the simplest advanced
    technique and provides the most consistent recall improvement.

    Parameters
    ----------
    base_retriever:
        Callable ``(query: str, top_k: int) -> list[RetrievedChunk]``.
    generator:
        Object with a ``paraphrase(query, n) -> list[str]`` method.
    num_variants:
        How many query paraphrases to generate (in addition to the original).
    config:
        Shared configuration.
    """

    def __init__(
        self,
        base_retriever: Any,
        generator: Any | None = None,
        num_variants: int = 3,
        config: AdvancedRetrievalConfig | None = None,
    ) -> None:
        """Initialise Multi-Query retriever with a base retriever and variant count."""
        super().__init__(config)
        self.base_retriever = base_retriever
        self.generator = generator or RuleBasedQueryGenerator()
        self.num_variants = num_variants

    def retrieve(self, query: str, *, top_k: int | None = None) -> AdvancedRetrievalResult:
        """Retrieve with multiple query variants fused via RRF."""
        k = top_k or self.config.top_k
        candidate_k = k * self.config.overretrieve_factor

        # Generate paraphrases
        paraphrases = self.generator.paraphrase(query, n=self.num_variants)
        all_queries = [query, *paraphrases]
        logger.debug("Multi-query variants: %s", all_queries)

        # Retrieve for each
        ranked_lists: list[list[RetrievedChunk]] = []
        for q in all_queries:
            try:
                chunks = self._fetch(q, candidate_k)
                ranked_lists.append(chunks)
            except Exception:
                logger.warning("Sub-query retrieval failed for: %r", q)

        if not ranked_lists:
            return AdvancedRetrievalResult(
                query=query,
                chunks=[],
                sub_queries=all_queries,
                strategy="multi_query",
            )

        # Fuse via RRF
        fused = reciprocal_rank_fusion(ranked_lists, k=self.config.rrf_k, top_n=k)
        deduped = self._dedup(fused)

        return AdvancedRetrievalResult(
            query=query,
            chunks=deduped[:k],
            sub_queries=all_queries,
            strategy="multi_query",
            metadata={"num_variants": len(paraphrases)},
        )

    def _fetch(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raw = self.base_retriever(query, top_k)
        return HyDERetriever._normalise(raw, source_query=query)


# ---------------------------------------------------------------------------
# Composite: chain multiple strategies
# ---------------------------------------------------------------------------


class CompositeAdvancedRetriever(BaseAdvancedRetriever):
    """Chain multiple advanced retrieval strategies and fuse their results.

    Parameters
    ----------
    retrievers:
        Ordered list of ``BaseAdvancedRetriever`` instances.  Each is called
        in sequence and results are fused via RRF.
    config:
        Controls the final ``top_k`` and RRF constant.
    """

    def __init__(
        self,
        retrievers: list[BaseAdvancedRetriever],
        config: AdvancedRetrievalConfig | None = None,
    ) -> None:
        """Initialise composite retriever with a list of child retrievers."""
        super().__init__(config)
        self.retrievers = retrievers

    def retrieve(self, query: str, *, top_k: int | None = None) -> AdvancedRetrievalResult:
        """Run all child retrievers and fuse their results."""
        k = top_k or self.config.top_k
        all_sub_queries: list[str] = []
        ranked_lists: list[list[RetrievedChunk]] = []

        for retriever in self.retrievers:
            try:
                result = retriever.retrieve(query, top_k=k * self.config.overretrieve_factor)
                ranked_lists.append(result.chunks)
                all_sub_queries.extend(result.sub_queries)
            except Exception:
                logger.warning("Composite sub-retriever %s failed", type(retriever).__name__)

        if not ranked_lists:
            return AdvancedRetrievalResult(
                query=query,
                chunks=[],
                sub_queries=all_sub_queries,
                strategy="composite",
            )

        fused = reciprocal_rank_fusion(ranked_lists, k=self.config.rrf_k, top_n=k)
        deduped = self._dedup(fused)

        return AdvancedRetrievalResult(
            query=query,
            chunks=deduped[:k],
            sub_queries=list(dict.fromkeys(all_sub_queries)),  # dedup while preserving order
            strategy="composite",
        )
