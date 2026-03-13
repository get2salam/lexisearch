"""Query intent classification and strategy routing for LexiSearch.

Analyses an incoming query, infers its *intent*, and selects (or constructs)
the most suitable retrieval strategy.  Intent classification is deliberately
rule-based — no LLM dependency — so it works offline and adds zero latency.

Intent taxonomy
---------------
FACTUAL
    Specific, narrow queries expecting a concrete answer.
    e.g. "What is the penalty for contempt of court?"
DEFINITIONAL
    Queries asking for definitions or explanations of terms/concepts.
    e.g. "Define res judicata" / "What does force majeure mean?"
PROCEDURAL
    Queries about processes, steps, or how something is done.
    e.g. "How do I file an appeal?" / "Procedure for winding up a company"
COMPARATIVE
    Queries comparing two or more concepts, statutes, or cases.
    e.g. "Difference between void and voidable contracts"
MULTI_HOP
    Complex queries requiring synthesis across multiple documents.
    e.g. "Landmark cases on constitutional freedom of expression since 2000"
UNKNOWN
    Catch-all when no intent can be reliably inferred.

Strategy mapping (defaults, all overridable)
--------------------------------------------
FACTUAL      → HyDE (hypothetical doc anchors to specific answer regions)
DEFINITIONAL → HyDE (hypothetical definition matches glossary chunks well)
PROCEDURAL   → StepBack (broader context gives procedural background)
COMPARATIVE  → MultiQuery (multiple phrasings surface both sides)
MULTI_HOP    → Composite[MultiQuery + StepBack]
UNKNOWN      → MultiQuery (safest general fallback)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------


class QueryIntent(str, Enum):
    """Enumeration of recognised query intent classes."""

    FACTUAL = "factual"
    DEFINITIONAL = "definitional"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi_hop"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------


@dataclass
class IntentClassification:
    """Result of classifying a single query."""

    query: str
    intent: QueryIntent
    confidence: float
    """Confidence in [0.0, 1.0].  Rule-based classifier returns 0.9 on a strong
    match, 0.6 on a weak heuristic match, 0.0 for UNKNOWN."""
    matched_patterns: list[str] = field(default_factory=list)
    """Human-readable description of which patterns fired."""


# ---------------------------------------------------------------------------
# Rule-based intent classifier
# ---------------------------------------------------------------------------


class IntentClassifier:
    """Classify query intent using regex pattern matching.

    All patterns are case-insensitive and matched against the lowercased query.
    Priority order (highest first): DEFINITIONAL, PROCEDURAL, COMPARATIVE,
    MULTI_HOP, FACTUAL.  UNKNOWN is the fallback.

    The classifier is intentionally conservative: it returns UNKNOWN rather
    than guessing, so the router falls back to the safest strategy.
    """

    _DEFINITIONAL_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (
            re.compile(r"\b(define|definition of|what (is|are|does)\b.{0,20}\bmean)\b"),
            "definition trigger",
        ),
        (re.compile(r"\bmeaning of\b"), "meaning-of"),
        (re.compile(r"\bexplain (the )?(concept|term|doctrine|principle)\b"), "explain-concept"),
        (re.compile(r"\bwhat is (a |an |the )?\w+\??$"), "what-is short"),
    ]

    _PROCEDURAL_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(r"\bhow (do|does|can|should|to)\b"), "how-to"),
        (re.compile(r"\bprocedure (for|to|of)\b"), "procedure-for"),
        (re.compile(r"\bsteps? (to|for|in)\b"), "steps-to"),
        (re.compile(r"\bprocess (of|for|to)\b"), "process-of"),
        (re.compile(r"\bfile (a|an|the)\b"), "file-a"),
        (re.compile(r"\bapply (for|to)\b"), "apply-for"),
    ]

    _COMPARATIVE_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(r"\b(difference|distinguish|compare|contrast)\b"), "compare-trigger"),
        (re.compile(r"\bvs\.?\b|\bversus\b"), "vs"),
        (re.compile(r"\b(similar(ity)?|dissimilar(ity)?)\b"), "similarity"),
        (re.compile(r"\bwhat.{0,30}(differ|distinction)\b"), "what-differs"),
    ]

    _MULTI_HOP_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (
            re.compile(
                r"\b(all|every|list (of|all))\b.*\b(cases?|statutes?|rulings?|judgments?)\b"
            ),
            "list-all",
        ),
        (re.compile(r"\blandmark\b"), "landmark"),
        (re.compile(r"\bsince \d{4}\b|\bafter \d{4}\b|\bbefore \d{4}\b"), "temporal range"),
        (re.compile(r"\band.{1,30}and\b"), "multi-and conjunctions"),
        (re.compile(r"\bmultiple\b.{0,20}\b(courts?|jurisdictions?|statutes?)\b"), "multi-source"),
    ]

    _FACTUAL_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(r"\bwhat (is|are|was|were)\b"), "what-is/was"),
        (re.compile(r"\bwho (is|was|can|must|shall)\b"), "who-is"),
        (re.compile(r"\bwhen (is|was|does|did|can|must)\b"), "when"),
        (re.compile(r"\bwhere (is|are|can|must)\b"), "where"),
        (re.compile(r"\b(penalty|punishment|sentence|fine) (for|of|under)\b"), "penalty-for"),
        (
            re.compile(r"\bmaximum|minimum\b.{0,20}\b(term|sentence|fine|penalty)\b"),
            "max/min sentence",
        ),
    ]

    def classify(self, query: str) -> IntentClassification:
        """Return the intent and confidence for *query*."""
        q = query.strip().lower()

        # Priority order matters — check most specific first
        for intent, patterns in [
            (QueryIntent.DEFINITIONAL, self._DEFINITIONAL_PATTERNS),
            (QueryIntent.PROCEDURAL, self._PROCEDURAL_PATTERNS),
            (QueryIntent.COMPARATIVE, self._COMPARATIVE_PATTERNS),
            (QueryIntent.MULTI_HOP, self._MULTI_HOP_PATTERNS),
            (QueryIntent.FACTUAL, self._FACTUAL_PATTERNS),
        ]:
            matched = [label for pat, label in patterns if pat.search(q)]
            if matched:
                confidence = 0.9 if len(matched) >= 2 else 0.6
                return IntentClassification(
                    query=query,
                    intent=intent,
                    confidence=confidence,
                    matched_patterns=matched,
                )

        return IntentClassification(
            query=query,
            intent=QueryIntent.UNKNOWN,
            confidence=0.0,
            matched_patterns=[],
        )


# ---------------------------------------------------------------------------
# Routing result
# ---------------------------------------------------------------------------


@dataclass
class RoutingResult:
    """Result of routing a query to a retrieval strategy."""

    query: str
    intent: QueryIntent
    confidence: float
    strategy_name: str
    retriever: Any
    """The constructed retriever instance, ready to call ``.retrieve()``."""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class QueryRouter:
    """Route queries to retrieval strategies based on inferred intent.

    Parameters
    ----------
    base_retriever:
        Callable ``(query: str, top_k: int) -> list[RetrievedChunk]`` used as
        the underlying retriever for all strategy wrappers.
    classifier:
        Optional custom intent classifier.  Defaults to
        :class:`IntentClassifier`.
    strategy_overrides:
        Map ``QueryIntent → callable`` to override default strategy factories.
        Each callable receives ``(base_retriever, config)`` and must return a
        retriever with a ``.retrieve(query, *, top_k)`` method.
    config:
        Shared :class:`~lexisearch.retrieval.advanced.AdvancedRetrievalConfig`.

    Example:
    --------
    ::

        router = QueryRouter(base_retriever=my_retriever)
        result = router.route("How do I appeal a judgment?")
        chunks = result.retriever.retrieve(result.query, top_k=5).chunks
    """

    def __init__(
        self,
        base_retriever: Any,
        classifier: IntentClassifier | None = None,
        strategy_overrides: dict[QueryIntent, Any] | None = None,
        config: Any | None = None,
    ) -> None:
        """Initialise the router with a base retriever and optional overrides."""
        # Import here to avoid circular imports at module level
        from lexisearch.retrieval.advanced import (
            AdvancedRetrievalConfig,
            CompositeAdvancedRetriever,
            HyDERetriever,
            MultiQueryRetriever,
            StepBackRetriever,
        )

        self.base_retriever = base_retriever
        self.classifier = classifier or IntentClassifier()
        self._cfg = config or AdvancedRetrievalConfig()
        self._overrides = strategy_overrides or {}

        # Store references to strategy classes for default factory
        self._HyDE = HyDERetriever
        self._StepBack = StepBackRetriever
        self._MultiQuery = MultiQueryRetriever
        self._Composite = CompositeAdvancedRetriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, query: str) -> RoutingResult:
        """Classify *query* and return the recommended retriever."""
        classification = self.classifier.classify(query)
        intent = classification.intent
        logger.debug(
            "Routing query %r → intent=%s (confidence=%.2f, patterns=%s)",
            query[:60],
            intent.value,
            classification.confidence,
            classification.matched_patterns,
        )

        retriever, strategy_name = self._build_strategy(intent)
        return RoutingResult(
            query=query,
            intent=intent,
            confidence=classification.confidence,
            strategy_name=strategy_name,
            retriever=retriever,
            metadata={
                "matched_patterns": classification.matched_patterns,
                "confidence": classification.confidence,
            },
        )

    def route_and_retrieve(self, query: str, *, top_k: int = 5) -> Any:
        """Route *query* and immediately call ``.retrieve()`` on the strategy.

        Returns an :class:`~lexisearch.retrieval.advanced.AdvancedRetrievalResult`
        augmented with routing metadata under ``result.metadata["routing"]``.
        """
        routing = self.route(query)
        result = routing.retriever.retrieve(query, top_k=top_k)
        result.metadata["routing"] = {
            "intent": routing.intent.value,
            "confidence": routing.confidence,
            "strategy": routing.strategy_name,
            "matched_patterns": routing.metadata.get("matched_patterns", []),
        }
        return result

    # ------------------------------------------------------------------
    # Internal: strategy factory
    # ------------------------------------------------------------------

    def _build_strategy(self, intent: QueryIntent) -> tuple[Any, str]:
        """Return ``(retriever_instance, strategy_name)`` for *intent*."""
        if intent in self._overrides:
            factory = self._overrides[intent]
            retriever = factory(self.base_retriever, self._cfg)
            return retriever, f"custom:{intent.value}"

        br = self.base_retriever
        cfg = self._cfg

        if intent in (QueryIntent.FACTUAL, QueryIntent.DEFINITIONAL):
            return self._HyDE(br, config=cfg), "hyde"

        if intent == QueryIntent.PROCEDURAL:
            return self._StepBack(br, config=cfg), "step_back"

        if intent == QueryIntent.COMPARATIVE:
            return self._MultiQuery(br, num_variants=4, config=cfg), "multi_query"

        if intent == QueryIntent.MULTI_HOP:
            mq = self._MultiQuery(br, num_variants=3, config=cfg)
            sb = self._StepBack(br, config=cfg)
            return self._Composite([mq, sb], config=cfg), "composite[multi_query+step_back]"

        # UNKNOWN → safest general strategy
        return self._MultiQuery(br, num_variants=3, config=cfg), "multi_query"
