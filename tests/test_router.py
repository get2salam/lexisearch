"""Tests for QueryRouter and IntentClassifier."""

from __future__ import annotations

import pytest

from lexisearch.retrieval.advanced import (
    AdvancedRetrievalConfig,
    CompositeAdvancedRetriever,
    HyDERetriever,
    MultiQueryRetriever,
    RetrievedChunk,
    StepBackRetriever,
)
from lexisearch.retrieval.router import (
    IntentClassification,
    IntentClassifier,
    QueryIntent,
    QueryRouter,
    RoutingResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_retriever(query: str, top_k: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            content=f"Content {i} for {query}",
            score=1.0 / (i + 1),
        )
        for i in range(top_k)
    ]


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------


class TestIntentClassifier:
    def setup_method(self):
        self.clf = IntentClassifier()

    def _classify(self, query: str) -> IntentClassification:
        return self.clf.classify(query)

    # Definitional
    @pytest.mark.parametrize(
        "query",
        [
            "Define res judicata",
            "What does force majeure mean?",
            "Explain the concept of promissory estoppel",
            "Definition of stare decisis",
        ],
    )
    def test_definitional_queries(self, query):
        result = self._classify(query)
        assert result.intent == QueryIntent.DEFINITIONAL
        assert result.confidence > 0.0

    # Procedural
    @pytest.mark.parametrize(
        "query",
        [
            "How do I file an appeal?",
            "How does the winding-up procedure work?",
            "What are the steps to register a trademark?",
            "Procedure for challenging an administrative decision",
            "Process of applying for bail",
        ],
    )
    def test_procedural_queries(self, query):
        result = self._classify(query)
        assert result.intent == QueryIntent.PROCEDURAL

    # Comparative
    @pytest.mark.parametrize(
        "query",
        [
            "Difference between void and voidable contracts",
            "Compare civil and criminal liability",
            "Distinguish between a lease and a licence",
            "Void vs voidable agreements",
        ],
    )
    def test_comparative_queries(self, query):
        result = self._classify(query)
        assert result.intent == QueryIntent.COMPARATIVE

    # Multi-hop
    @pytest.mark.parametrize(
        "query",
        [
            "List all landmark cases on free speech since 2000",
            "All statutes governing data protection after 2010",
        ],
    )
    def test_multi_hop_queries(self, query):
        result = self._classify(query)
        assert result.intent == QueryIntent.MULTI_HOP

    # Factual
    @pytest.mark.parametrize(
        "query",
        [
            "What is the maximum sentence for contempt of court?",
            "When does a contract become void?",
            "Who was the judge in that case?",
        ],
    )
    def test_factual_queries(self, query):
        result = self._classify(query)
        assert result.intent == QueryIntent.FACTUAL

    def test_unknown_query(self):
        result = self._classify("xyzzy")
        assert result.intent == QueryIntent.UNKNOWN
        assert result.confidence == 0.0
        assert result.matched_patterns == []

    def test_high_confidence_on_multiple_pattern_match(self):
        # "definition of" + "meaning of" should both fire
        result = self._classify("Definition of promissory estoppel and meaning of waiver")
        assert result.confidence == 0.9

    def test_classification_result_fields(self):
        result = self._classify("How do I apply for a stay order?")
        assert result.query == "How do I apply for a stay order?"
        assert isinstance(result.matched_patterns, list)
        assert len(result.matched_patterns) >= 1


# ---------------------------------------------------------------------------
# QueryRouter — strategy selection
# ---------------------------------------------------------------------------


class TestQueryRouter:
    def setup_method(self):
        self.cfg = AdvancedRetrievalConfig(top_k=3, overretrieve_factor=2)
        self.router = QueryRouter(base_retriever=_stub_retriever, config=self.cfg)

    def test_returns_routing_result(self):
        result = self.router.route("What is habeas corpus?")
        assert isinstance(result, RoutingResult)
        assert result.query == "What is habeas corpus?"

    def test_definitional_routes_to_hyde(self):
        result = self.router.route("Define mens rea")
        assert isinstance(result.retriever, HyDERetriever)
        assert result.strategy_name == "hyde"
        assert result.intent == QueryIntent.DEFINITIONAL

    def test_factual_routes_to_hyde(self):
        result = self.router.route("What is the penalty for contempt?")
        assert isinstance(result.retriever, HyDERetriever)
        assert result.intent == QueryIntent.FACTUAL

    def test_procedural_routes_to_stepback(self):
        result = self.router.route("How do I file an appeal in court?")
        assert isinstance(result.retriever, StepBackRetriever)
        assert result.strategy_name == "step_back"

    def test_comparative_routes_to_multiquery(self):
        result = self.router.route("Difference between void and voidable contracts")
        assert isinstance(result.retriever, MultiQueryRetriever)
        assert result.intent == QueryIntent.COMPARATIVE

    def test_multi_hop_routes_to_composite(self):
        result = self.router.route("All landmark cases on freedom of expression since 1990")
        assert isinstance(result.retriever, CompositeAdvancedRetriever)
        assert "composite" in result.strategy_name

    def test_unknown_routes_to_multiquery(self):
        result = self.router.route("xyzzy plugh")
        assert isinstance(result.retriever, MultiQueryRetriever)
        assert result.intent == QueryIntent.UNKNOWN

    def test_routing_metadata(self):
        result = self.router.route("Define promissory estoppel")
        assert "matched_patterns" in result.metadata
        assert "confidence" in result.metadata

    def test_custom_override(self):
        custom_retriever = MultiQueryRetriever(_stub_retriever, config=self.cfg)

        def factory(base_retriever, config):
            return custom_retriever

        router = QueryRouter(
            base_retriever=_stub_retriever,
            strategy_overrides={QueryIntent.DEFINITIONAL: factory},
            config=self.cfg,
        )
        result = router.route("Define estoppel")
        assert result.retriever is custom_retriever
        assert "custom:" in result.strategy_name


# ---------------------------------------------------------------------------
# route_and_retrieve integration
# ---------------------------------------------------------------------------


class TestRouteAndRetrieve:
    def setup_method(self):
        self.router = QueryRouter(
            base_retriever=_stub_retriever,
            config=AdvancedRetrievalConfig(top_k=3, overretrieve_factor=2),
        )

    def test_returns_advanced_retrieval_result(self):
        from lexisearch.retrieval.advanced import AdvancedRetrievalResult

        result = self.router.route_and_retrieve("What is consideration in contract law?", top_k=3)
        assert isinstance(result, AdvancedRetrievalResult)
        assert len(result.chunks) <= 3

    def test_routing_metadata_injected(self):
        result = self.router.route_and_retrieve("How do I appeal?", top_k=2)
        routing_meta = result.metadata.get("routing", {})
        assert "intent" in routing_meta
        assert "strategy" in routing_meta
        assert "confidence" in routing_meta

    def test_chunks_have_content(self):
        result = self.router.route_and_retrieve("Define stare decisis", top_k=3)
        for chunk in result.chunks:
            assert chunk.content.strip()
