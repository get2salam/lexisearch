"""BM25 sparse retriever implementation.

Implements the Okapi BM25 ranking function for lexical/keyword-based retrieval.
This provides the sparse component for hybrid search pipelines.

References:
    Robertson, S. E., & Zaragoza, H. (2009). The Probabilistic Relevance
    Framework: BM25 and Beyond. *Foundations and Trends in Information
    Retrieval*, 3(4), 333-389.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any

from lexisearch.models import Chunk, SearchResult
from lexisearch.retrieval.base import (
    BaseRetriever,
    MetadataFilter,
    RetrieverConfig,
    RetrieverType,
)

# Default English stop words (minimal set for broad applicability)
_DEFAULT_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "it",
        "was",
        "are",
        "were",
        "been",
        "be",
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
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "as",
        "up",
        "out",
        "about",
    }
)

_PUNCTUATION_RE = re.compile(f"[{re.escape(string.punctuation)}]")


@dataclass
class BM25Config:
    """Configuration for BM25 scoring.

    Attributes:
        k1: Term frequency saturation parameter. Higher values give more
            weight to term frequency. Typical range: 1.2 - 2.0.
        b: Length normalisation parameter. 0 disables normalisation,
            1.0 fully normalises by document length.
        epsilon: Floor value for IDF to avoid negative scores for very
            common terms.
        stop_words: Set of stop words to filter out during tokenisation.
        lowercase: Whether to lowercase tokens.
        strip_punctuation: Whether to remove punctuation before tokenising.
    """

    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    stop_words: frozenset[str] = _DEFAULT_STOP_WORDS
    lowercase: bool = True
    strip_punctuation: bool = True


@dataclass
class _IndexedDocument:
    """Internal representation of an indexed document for BM25 scoring.

    Attributes:
        chunk: The original chunk.
        tokens: Tokenised content.
        term_freq: Term frequency counter for the document.
        length: Number of tokens.
    """

    chunk: Chunk
    tokens: list[str]
    term_freq: Counter[str]
    length: int


class BM25Retriever(BaseRetriever):
    """BM25 (Okapi) sparse retriever for keyword-based document ranking.

    Maintains an in-memory inverted index over :class:`Chunk` objects.
    Supports incremental document addition and removal.

    Args:
        config: General retriever configuration.
        bm25_config: BM25-specific parameters.

    Example::

        retriever = BM25Retriever()
        retriever.add_chunks(chunks)
        results = retriever.retrieve("information retrieval", top_k=5)
    """

    def __init__(
        self,
        config: RetrieverConfig | None = None,
        bm25_config: BM25Config | None = None,
    ) -> None:
        """Initialize BM25Retriever."""
        super().__init__(config)
        self.bm25_config = bm25_config or BM25Config()

        # Corpus storage
        self._documents: dict[str, _IndexedDocument] = {}
        # Inverted index: term → set of chunk IDs containing the term
        self._inverted_index: dict[str, set[str]] = {}
        # Corpus statistics
        self._total_tokens: int = 0
        self._avg_doc_length: float = 0.0

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms for BM25 scoring.

        Applies lowercasing, punctuation removal, and stop-word filtering
        based on the :class:`BM25Config`.

        Args:
            text: Raw input text.

        Returns:
            List of filtered tokens.
        """
        if self.bm25_config.lowercase:
            text = text.lower()
        if self.bm25_config.strip_punctuation:
            text = _PUNCTUATION_RE.sub(" ", text)

        tokens = text.split()

        if self.bm25_config.stop_words:
            tokens = [t for t in tokens if t not in self.bm25_config.stop_words]

        return tokens

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the BM25 index.

        Args:
            chunks: Chunks to index.

        Returns:
            Number of new chunks added (duplicates are skipped).
        """
        added = 0
        for chunk in chunks:
            if chunk.id in self._documents:
                continue

            tokens = self.tokenize(chunk.content)
            term_freq = Counter(tokens)
            doc = _IndexedDocument(
                chunk=chunk,
                tokens=tokens,
                term_freq=term_freq,
                length=len(tokens),
            )
            self._documents[chunk.id] = doc
            self._total_tokens += doc.length

            # Update inverted index
            for term in term_freq:
                if term not in self._inverted_index:
                    self._inverted_index[term] = set()
                self._inverted_index[term].add(chunk.id)

            added += 1

        # Recompute average document length
        if self._documents:
            self._avg_doc_length = self._total_tokens / len(self._documents)

        return added

    def remove_chunks(self, ids: list[str]) -> int:
        """Remove chunks from the BM25 index by ID.

        Args:
            ids: Chunk IDs to remove.

        Returns:
            Number of chunks actually removed.
        """
        removed = 0
        for chunk_id in ids:
            doc = self._documents.pop(chunk_id, None)
            if doc is None:
                continue

            self._total_tokens -= doc.length
            removed += 1

            # Clean inverted index
            for term in doc.term_freq:
                term_set = self._inverted_index.get(term)
                if term_set:
                    term_set.discard(chunk_id)
                    if not term_set:
                        del self._inverted_index[term]

        # Recompute average
        if self._documents:
            self._avg_doc_length = self._total_tokens / len(self._documents)
        else:
            self._avg_doc_length = 0.0

        return removed

    def clear(self) -> None:
        """Remove all documents from the index."""
        self._documents.clear()
        self._inverted_index.clear()
        self._total_tokens = 0
        self._avg_doc_length = 0.0

    @property
    def corpus_size(self) -> int:
        """Return the number of indexed documents."""
        return len(self._documents)

    # ------------------------------------------------------------------
    # BM25 scoring
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        r"""Compute the Inverse Document Frequency for a term.

        Uses the standard BM25 IDF formula with an epsilon floor:

        .. math::
            \\text{IDF}(t) = \\ln\\!\\left(\\frac{N - n(t) + 0.5}{n(t) + 0.5}\\right)

        Args:
            term: The query term.

        Returns:
            IDF score (floored at ``epsilon``).
        """
        n = len(self._inverted_index.get(term, set()))
        total_docs = len(self._documents)
        idf = math.log((total_docs - n + 0.5) / (n + 0.5) + 1.0)
        return max(idf, self.bm25_config.epsilon)

    def score_document(self, query_tokens: list[str], doc_id: str) -> float:
        """Compute the BM25 score for a single document against query tokens.

        Args:
            query_tokens: Tokenised query terms.
            doc_id: The document chunk ID.

        Returns:
            BM25 relevance score.
        """
        doc = self._documents.get(doc_id)
        if doc is None:
            return 0.0

        k1 = self.bm25_config.k1
        b = self.bm25_config.b
        avg_dl = self._avg_doc_length or 1.0

        score = 0.0
        for term in query_tokens:
            tf = doc.term_freq.get(term, 0)
            if tf == 0:
                continue

            idf = self._idf(term)
            # BM25 term frequency normalisation
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc.length / avg_dl)
            score += idf * (numerator / denominator)

        return score

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
        """Retrieve chunks ranked by BM25 relevance.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results.
            filters: Metadata filters to apply.
            **kwargs: Unused.

        Returns:
            Ordered list of :class:`SearchResult` (highest BM25 score first).
        """
        k = top_k or self.config.top_k
        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        # Find candidate documents (any doc containing at least one query term)
        candidate_ids: set[str] = set()
        for term in query_tokens:
            candidate_ids.update(self._inverted_index.get(term, set()))

        # Score candidates
        scored: list[tuple[str, float]] = []
        for doc_id in candidate_ids:
            doc = self._documents[doc_id]

            # Apply metadata filters
            if filters and not self.apply_metadata_filter(doc.chunk, filters):
                continue

            score = self.score_document(query_tokens, doc_id)
            if score > 0:
                scored.append((doc_id, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results: list[SearchResult] = []
        for rank, (doc_id, score) in enumerate(scored[:k], start=1):
            doc = self._documents[doc_id]
            results.append(
                SearchResult(
                    chunk=doc.chunk,
                    score=score,
                    rank=rank,
                    metadata={"retriever": "bm25", "query_tokens": query_tokens},
                )
            )

        return results

    def retriever_type(self) -> RetrieverType:
        """Return the retriever type.

        Returns:
            :attr:`RetrieverType.SPARSE`
        """
        return RetrieverType.SPARSE

    def get_term_stats(self) -> dict[str, Any]:
        """Return corpus-level term statistics.

        Returns:
            Dictionary with vocabulary size, corpus size, and average
            document length.
        """
        return {
            "vocabulary_size": len(self._inverted_index),
            "corpus_size": len(self._documents),
            "avg_doc_length": round(self._avg_doc_length, 2),
            "total_tokens": self._total_tokens,
        }
