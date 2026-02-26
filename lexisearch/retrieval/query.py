"""Query expansion and transformation utilities.

Provides strategies for improving retrieval recall by expanding or
reformulating the original query before it reaches the retriever.

Strategies:
    - **Synonym expansion:** Augment query with known synonyms.
    - **Query decomposition:** Split complex queries into sub-queries.
    - **Pseudo-relevance feedback (PRF):** Use terms from top results to
      expand the query.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExpandedQuery:
    """Result of a query expansion operation.

    Attributes:
        original: The original query string.
        expanded: The expanded query string.
        sub_queries: Individual sub-queries (for decomposition strategies).
        added_terms: Terms added during expansion.
        strategy: Name of the expansion strategy used.
        metadata: Additional strategy-specific information.
    """

    original: str
    expanded: str
    sub_queries: list[str] = field(default_factory=list)
    added_terms: list[str] = field(default_factory=list)
    strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseQueryExpander(ABC):
    """Abstract base class for query expansion strategies."""

    @abstractmethod
    def expand(self, query: str) -> ExpandedQuery:
        """Expand or transform a query.

        Args:
            query: The original query string.

        Returns:
            An :class:`ExpandedQuery` with the expanded form.
        """
        ...


class SynonymExpander(BaseQueryExpander):
    """Expand queries with synonym terms from a configurable dictionary.

    Args:
        synonyms: Mapping from terms to their synonyms.
        max_synonyms_per_term: Maximum synonyms to add per matched term.
        case_sensitive: Whether to match terms case-sensitively.

    Example::

        expander = SynonymExpander(
            synonyms={"ML": ["machine learning"], "AI": ["artificial intelligence"]}
        )
        result = expander.expand("ML and AI techniques")
        # result.expanded includes original + synonym terms
    """

    def __init__(
        self,
        synonyms: dict[str, list[str]] | None = None,
        max_synonyms_per_term: int = 3,
        case_sensitive: bool = False,
    ) -> None:
        """Initialize SynonymExpander."""
        self.synonyms = synonyms or {}
        self.max_synonyms_per_term = max_synonyms_per_term
        self.case_sensitive = case_sensitive

        # Build lookup index
        self._index: dict[str, list[str]] = {}
        for term, syns in self.synonyms.items():
            key = term if case_sensitive else term.lower()
            self._index[key] = syns[: self.max_synonyms_per_term]

    def expand(self, query: str) -> ExpandedQuery:
        """Expand query with synonym terms.

        Args:
            query: Original query.

        Returns:
            Expanded query with synonym terms appended.
        """
        tokens = query.split()
        added: list[str] = []

        for token in tokens:
            key = token if self.case_sensitive else token.lower()
            # Strip punctuation for matching
            clean_key = re.sub(r"[^\w]", "", key)
            syns = self._index.get(clean_key, [])
            for syn in syns:
                if syn.lower() not in query.lower():
                    added.append(syn)

        expanded = query
        if added:
            expanded = f"{query} {' '.join(added)}"

        return ExpandedQuery(
            original=query,
            expanded=expanded,
            added_terms=added,
            strategy="synonym",
        )


class QueryDecomposer(BaseQueryExpander):
    """Decompose complex queries into simpler sub-queries.

    Splits queries on conjunctions and punctuation to create focused
    sub-queries, each of which can be independently retrieved.

    Args:
        split_patterns: Regex patterns to split on.
        min_sub_query_length: Minimum character length for a sub-query.

    Example::

        decomposer = QueryDecomposer()
        result = decomposer.expand(
            "What are transformers and how do they compare to RNNs?"
        )
        # result.sub_queries might contain:
        # ["What are transformers", "how do they compare to RNNs"]
    """

    def __init__(
        self,
        split_patterns: list[str] | None = None,
        min_sub_query_length: int = 5,
    ) -> None:
        """Initialize the instance."""
        self.split_patterns = split_patterns or [
            r"\band\b",
            r"\bor\b",
            r"\bbut\b",
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\bcompare[d]?\s+(?:to|with)\b",
            r"[;]",
        ]
        self.min_sub_query_length = min_sub_query_length
        self._pattern = re.compile(
            "|".join(f"({p})" for p in self.split_patterns),
            re.IGNORECASE,
        )

    def expand(self, query: str) -> ExpandedQuery:
        """Decompose a complex query into sub-queries.

        Args:
            query: Original query.

        Returns:
            ExpandedQuery with sub_queries populated.
        """
        parts = self._pattern.split(query)
        sub_queries: list[str] = []

        for part in parts:
            if part is None:
                continue
            cleaned = part.strip()
            # Skip the separator tokens themselves
            if self._pattern.fullmatch(cleaned):
                continue
            if len(cleaned) >= self.min_sub_query_length:
                sub_queries.append(cleaned)

        # If decomposition produced nothing useful, use the original
        if not sub_queries:
            sub_queries = [query]

        return ExpandedQuery(
            original=query,
            expanded=query,
            sub_queries=sub_queries,
            strategy="decomposition",
            metadata={"num_sub_queries": len(sub_queries)},
        )


class PseudoRelevanceFeedback(BaseQueryExpander):
    """Expand queries using terms from pseudo-relevant documents.

    Given a retriever, runs the original query, extracts frequent terms
    from the top results, and appends them to the query. This
    implements Rocchio-style blind relevance feedback.

    Args:
        retriever: A retriever to fetch initial results.
        num_feedback_docs: Number of top documents to use for feedback.
        num_expansion_terms: Number of terms to add from feedback documents.
        stop_words: Terms to exclude from expansion.
    """

    def __init__(
        self,
        retriever: Any,
        num_feedback_docs: int = 3,
        num_expansion_terms: int = 5,
        stop_words: frozenset[str] | None = None,
    ) -> None:
        """Initialize PseudoRelevanceFeedback."""
        self.retriever = retriever
        self.num_feedback_docs = num_feedback_docs
        self.num_expansion_terms = num_expansion_terms
        self.stop_words = stop_words or frozenset(
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
                "this",
                "that",
                "these",
                "those",
                "not",
                "no",
            }
        )

    def expand(self, query: str) -> ExpandedQuery:
        """Expand query using pseudo-relevance feedback.

        Retrieves top documents for the original query, extracts
        frequent terms, and appends the most common to the query.

        Args:
            query: Original query.

        Returns:
            Expanded query with feedback terms.
        """
        from collections import Counter

        # Retrieve initial results
        results = self.retriever.retrieve(query, top_k=self.num_feedback_docs)

        if not results:
            return ExpandedQuery(
                original=query,
                expanded=query,
                strategy="prf",
                metadata={"feedback_docs": 0},
            )

        # Extract and count terms from feedback documents
        query_terms = set(query.lower().split())
        term_counts: Counter[str] = Counter()

        for result in results:
            tokens = re.findall(r"\b\w+\b", result.chunk.content.lower())
            for token in tokens:
                if token not in self.stop_words and token not in query_terms and len(token) > 2:
                    term_counts[token] += 1

        # Select top expansion terms
        expansion_terms = [term for term, _ in term_counts.most_common(self.num_expansion_terms)]

        expanded = query
        if expansion_terms:
            expanded = f"{query} {' '.join(expansion_terms)}"

        return ExpandedQuery(
            original=query,
            expanded=expanded,
            added_terms=expansion_terms,
            strategy="prf",
            metadata={
                "feedback_docs": len(results),
                "candidate_terms": len(term_counts),
            },
        )


class MultiQueryExpander(BaseQueryExpander):
    """Generate multiple query variations for improved recall.

    Creates variations of the original query by applying simple
    transformations: keyword extraction, question reformulation,
    and term reordering.

    Args:
        max_variations: Maximum number of query variations to generate.
    """

    def __init__(self, max_variations: int = 3) -> None:
        """Initialize MultiQueryExpander."""
        self.max_variations = max_variations

    def expand(self, query: str) -> ExpandedQuery:
        """Generate query variations.

        Args:
            query: Original query.

        Returns:
            ExpandedQuery with sub_queries containing variations.
        """
        variations: list[str] = [query]

        # Variation 1: Extract key terms (remove question words and stop words)
        question_words = {
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "is",
            "are",
            "do",
            "does",
            "can",
            "could",
        }
        tokens = query.split()
        key_terms = [t for t in tokens if t.lower().strip("?.,!") not in question_words]
        if key_terms and len(key_terms) >= 2:
            variations.append(" ".join(key_terms))

        # Variation 2: Reverse term order (can help with different embeddings)
        if len(tokens) >= 3:
            reversed_terms = [
                t for t in reversed(tokens) if t.lower().strip("?.,!") not in question_words
            ]
            if reversed_terms:
                variations.append(" ".join(reversed_terms))

        # Variation 3: Remove question mark and restructure
        if query.rstrip().endswith("?"):
            declarative = query.rstrip("? ").strip()
            if declarative != query:
                variations.append(declarative)

        # Deduplicate and limit
        seen: set[str] = set()
        unique: list[str] = []
        for v in variations:
            normalised = v.strip().lower()
            if normalised not in seen:
                seen.add(normalised)
                unique.append(v.strip())

        unique = unique[: self.max_variations + 1]

        return ExpandedQuery(
            original=query,
            expanded=query,
            sub_queries=unique,
            strategy="multi_query",
            metadata={"num_variations": len(unique)},
        )
