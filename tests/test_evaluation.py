"""Tests for the LexiSearch evaluation module."""

from __future__ import annotations

import pytest

from lexisearch.evaluation import (
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    EvalReport,
    EvalSample,
    Evaluator,
    ExactMatchMetric,
    FaithfulnessMetric,
    MetricResult,
    SampleResult,
    TokenF1Metric,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_sample() -> EvalSample:
    """Return an EvalSample where answer exactly matches context and reference."""
    return EvalSample(
        question="What is retrieval-augmented generation?",
        contexts=[
            "Retrieval-augmented generation combines retrieval with generation.",
            "RAG uses retrieved documents to ground the language model response.",
        ],
        answer="Retrieval-augmented generation combines retrieval with generation.",
        reference="Retrieval-augmented generation combines retrieval with generation.",
        sample_id="perfect-001",
    )


@pytest.fixture
def partial_sample() -> EvalSample:
    """Return an EvalSample with a partially supported answer."""
    return EvalSample(
        question="What is machine learning?",
        contexts=["Machine learning is a method for teaching computers to learn."],
        answer="Machine learning is artificial intelligence.",
        reference="Machine learning teaches computers from data.",
    )


@pytest.fixture
def no_reference_sample() -> EvalSample:
    """Return an EvalSample without a reference answer."""
    return EvalSample(
        question="Explain embeddings.",
        contexts=["Embeddings are dense vector representations of text."],
        answer="Embeddings represent text as dense vectors.",
    )


# ---------------------------------------------------------------------------
# EvalSample tests
# ---------------------------------------------------------------------------


class TestEvalSample:
    def test_basic_creation(self) -> None:
        sample = EvalSample(
            question="Q",
            contexts=["C"],
            answer="A",
        )
        assert sample.question == "Q"
        assert sample.contexts == ["C"]
        assert sample.answer == "A"
        assert sample.reference is None
        assert sample.sample_id is None
        assert sample.metadata == {}

    def test_with_reference(self) -> None:
        sample = EvalSample(question="Q", contexts=[], answer="A", reference="R")
        assert sample.reference == "R"

    def test_with_metadata(self) -> None:
        sample = EvalSample(
            question="Q",
            contexts=[],
            answer="A",
            metadata={"source": "test"},
        )
        assert sample.metadata["source"] == "test"


# ---------------------------------------------------------------------------
# FaithfulnessMetric tests
# ---------------------------------------------------------------------------


class TestFaithfulnessMetric:
    def test_name(self) -> None:
        assert FaithfulnessMetric().name == "faithfulness"

    def test_perfect_faithfulness(self, perfect_sample: EvalSample) -> None:
        result = FaithfulnessMetric().score(perfect_sample)
        assert isinstance(result, MetricResult)
        assert result.score == pytest.approx(1.0, abs=0.05)
        assert result.metric_name == "faithfulness"

    def test_empty_answer(self) -> None:
        sample = EvalSample(
            question="Q",
            contexts=["some context here"],
            answer="",
        )
        result = FaithfulnessMetric().score(sample)
        assert result.score == pytest.approx(1.0)

    def test_zero_faithfulness(self) -> None:
        sample = EvalSample(
            question="Q",
            contexts=["apple banana cherry"],
            answer="xylophone zephyr quasar",
        )
        result = FaithfulnessMetric().score(sample)
        assert result.score == pytest.approx(0.0)

    def test_partial_faithfulness(self) -> None:
        sample = EvalSample(
            question="Q",
            contexts=["machine learning is powerful"],
            answer="machine learning is powerful but also xyzzy",
        )
        result = FaithfulnessMetric().score(sample)
        assert 0.0 < result.score < 1.0

    def test_details_populated(self, perfect_sample: EvalSample) -> None:
        result = FaithfulnessMetric().score(perfect_sample)
        assert "answer_tokens" in result.details
        assert "supported_tokens" in result.details
        assert "context_length" in result.details

    def test_score_in_range(self, partial_sample: EvalSample) -> None:
        result = FaithfulnessMetric().score(partial_sample)
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# ContextPrecisionMetric tests
# ---------------------------------------------------------------------------


class TestContextPrecisionMetric:
    def test_name(self) -> None:
        assert ContextPrecisionMetric().name == "context_precision"

    def test_empty_contexts(self) -> None:
        sample = EvalSample(question="Q", contexts=[], answer="A")
        result = ContextPrecisionMetric().score(sample)
        assert result.score == pytest.approx(0.0)

    def test_highly_relevant_contexts(self, perfect_sample: EvalSample) -> None:
        result = ContextPrecisionMetric().score(perfect_sample)
        assert result.score >= 0.5

    def test_irrelevant_contexts(self) -> None:
        sample = EvalSample(
            question="What is machine learning?",
            contexts=["The sky is blue today.", "Cats like milk."],
            answer="Machine learning is a type of AI.",
        )
        metric = ContextPrecisionMetric(relevance_threshold=0.2)
        result = metric.score(sample)
        assert result.score == pytest.approx(0.0)

    def test_details_populated(self, perfect_sample: EvalSample) -> None:
        result = ContextPrecisionMetric().score(perfect_sample)
        assert "total_contexts" in result.details
        assert "relevant_contexts" in result.details

    def test_custom_threshold(self) -> None:
        sample = EvalSample(
            question="cats",
            contexts=["cats dogs birds"],
            answer="cats",
        )
        strict = ContextPrecisionMetric(relevance_threshold=0.9)
        loose = ContextPrecisionMetric(relevance_threshold=0.01)
        strict_result = strict.score(sample)
        loose_result = loose.score(sample)
        assert loose_result.score >= strict_result.score

    def test_score_in_range(self, partial_sample: EvalSample) -> None:
        result = ContextPrecisionMetric().score(partial_sample)
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# ContextRecallMetric tests
# ---------------------------------------------------------------------------


class TestContextRecallMetric:
    def test_name(self) -> None:
        assert ContextRecallMetric().name == "context_recall"

    def test_perfect_recall(self, perfect_sample: EvalSample) -> None:
        result = ContextRecallMetric().score(perfect_sample)
        assert result.score == pytest.approx(1.0, abs=0.05)

    def test_empty_reference(self) -> None:
        sample = EvalSample(
            question="Q",
            contexts=["Some context."],
            answer="",
            reference="",
        )
        result = ContextRecallMetric().score(sample)
        assert result.score == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        sample = EvalSample(
            question="Q",
            contexts=["apple banana cherry"],
            answer="A",
            reference="xylophone zephyr quasar",
        )
        result = ContextRecallMetric().score(sample)
        assert result.score == pytest.approx(0.0)

    def test_uses_answer_when_no_reference(self, no_reference_sample: EvalSample) -> None:
        result = ContextRecallMetric().score(no_reference_sample)
        assert 0.0 <= result.score <= 1.0
        assert result.details["has_reference"] is False

    def test_details_populated(self, perfect_sample: EvalSample) -> None:
        result = ContextRecallMetric().score(perfect_sample)
        assert "reference_tokens" in result.details
        assert "recalled_tokens" in result.details
        assert "has_reference" in result.details


# ---------------------------------------------------------------------------
# AnswerRelevanceMetric tests
# ---------------------------------------------------------------------------


class TestAnswerRelevanceMetric:
    def test_name(self) -> None:
        assert AnswerRelevanceMetric().name == "answer_relevance"

    def test_identical_tokens(self) -> None:
        sample = EvalSample(
            question="machine learning",
            contexts=[],
            answer="machine learning",
        )
        result = AnswerRelevanceMetric().score(sample)
        assert result.score == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        sample = EvalSample(
            question="apple banana",
            contexts=[],
            answer="xylophone zephyr",
        )
        result = AnswerRelevanceMetric().score(sample)
        assert result.score == pytest.approx(0.0)

    def test_partial_overlap(self, partial_sample: EvalSample) -> None:
        result = AnswerRelevanceMetric().score(partial_sample)
        assert 0.0 <= result.score <= 1.0

    def test_details_populated(self, perfect_sample: EvalSample) -> None:
        result = AnswerRelevanceMetric().score(perfect_sample)
        assert "question_tokens" in result.details
        assert "answer_tokens" in result.details
        assert "shared_tokens" in result.details


# ---------------------------------------------------------------------------
# ExactMatchMetric tests
# ---------------------------------------------------------------------------


class TestExactMatchMetric:
    def test_name(self) -> None:
        assert ExactMatchMetric().name == "exact_match"

    def test_exact_match(self, perfect_sample: EvalSample) -> None:
        result = ExactMatchMetric().score(perfect_sample)
        assert result.score == pytest.approx(1.0)

    def test_no_reference(self, no_reference_sample: EvalSample) -> None:
        result = ExactMatchMetric().score(no_reference_sample)
        assert result.score == pytest.approx(0.0)
        assert "error" in result.details

    def test_case_insensitive(self) -> None:
        sample = EvalSample(
            question="Q",
            contexts=[],
            answer="Hello World",
            reference="hello world",
        )
        result = ExactMatchMetric().score(sample)
        assert result.score == pytest.approx(1.0)

    def test_punctuation_stripped(self) -> None:
        sample = EvalSample(
            question="Q",
            contexts=[],
            answer="Hello, world!",
            reference="hello world",
        )
        result = ExactMatchMetric().score(sample)
        assert result.score == pytest.approx(1.0)

    def test_no_match(self, partial_sample: EvalSample) -> None:
        result = ExactMatchMetric().score(partial_sample)
        assert result.score == pytest.approx(0.0)

    def test_normalize_helper(self) -> None:
        normalized = ExactMatchMetric._normalize("  Hello,  WORLD!  ")
        assert normalized == "hello world"


# ---------------------------------------------------------------------------
# TokenF1Metric tests
# ---------------------------------------------------------------------------


class TestTokenF1Metric:
    def test_name(self) -> None:
        assert TokenF1Metric().name == "token_f1"

    def test_perfect_f1(self, perfect_sample: EvalSample) -> None:
        result = TokenF1Metric().score(perfect_sample)
        assert result.score == pytest.approx(1.0)

    def test_no_reference(self, no_reference_sample: EvalSample) -> None:
        result = TokenF1Metric().score(no_reference_sample)
        assert result.score == pytest.approx(0.0)
        assert "error" in result.details

    def test_zero_f1(self) -> None:
        sample = EvalSample(
            question="Q",
            contexts=[],
            answer="apple banana",
            reference="xylophone zephyr",
        )
        result = TokenF1Metric().score(sample)
        assert result.score == pytest.approx(0.0)

    def test_partial_f1(self, partial_sample: EvalSample) -> None:
        result = TokenF1Metric().score(partial_sample)
        assert 0.0 <= result.score <= 1.0

    def test_details_populated(self, perfect_sample: EvalSample) -> None:
        result = TokenF1Metric().score(perfect_sample)
        assert "answer_tokens" in result.details
        assert "reference_tokens" in result.details


# ---------------------------------------------------------------------------
# BaseMetric helper method tests
# ---------------------------------------------------------------------------


class TestBaseMetricHelpers:
    def test_tokenize(self) -> None:
        tokens = FaithfulnessMetric._tokenize("Hello, World! 123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens

    def test_token_set(self) -> None:
        tokens = FaithfulnessMetric._token_set("hello hello world")
        assert tokens == {"hello", "world"}

    def test_f1_identical(self) -> None:
        f1 = FaithfulnessMetric._f1(["a", "b"], ["a", "b"])
        assert f1 == pytest.approx(1.0)

    def test_f1_disjoint(self) -> None:
        f1 = FaithfulnessMetric._f1(["a", "b"], ["c", "d"])
        assert f1 == pytest.approx(0.0)

    def test_f1_both_empty(self) -> None:
        assert FaithfulnessMetric._f1([], []) == pytest.approx(1.0)

    def test_f1_one_empty(self) -> None:
        assert FaithfulnessMetric._f1(["a"], []) == pytest.approx(0.0)

    def test_jaccard_identical(self) -> None:
        j = FaithfulnessMetric._jaccard({"a", "b"}, {"a", "b"})
        assert j == pytest.approx(1.0)

    def test_jaccard_disjoint(self) -> None:
        j = FaithfulnessMetric._jaccard({"a"}, {"b"})
        assert j == pytest.approx(0.0)

    def test_jaccard_both_empty(self) -> None:
        j = FaithfulnessMetric._jaccard(set(), set())
        assert j == pytest.approx(1.0)

    def test_jaccard_partial(self) -> None:
        j = FaithfulnessMetric._jaccard({"a", "b"}, {"b", "c"})
        # intersection={b}, union={a,b,c} -> 1/3
        assert j == pytest.approx(1 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_default_metrics(self) -> None:
        ev = Evaluator()
        names = {m.name for m in ev.metrics}
        assert "faithfulness" in names
        assert "context_precision" in names
        assert "context_recall" in names
        assert "answer_relevance" in names
        assert "token_f1" in names
        assert "exact_match" in names

    def test_custom_metrics(self) -> None:
        ev = Evaluator(metrics=[FaithfulnessMetric()])
        assert len(ev.metrics) == 1
        assert ev.metrics[0].name == "faithfulness"

    def test_add_metric_fluent(self) -> None:
        ev = Evaluator(metrics=[])
        result = ev.add_metric(FaithfulnessMetric()).add_metric(TokenF1Metric())
        assert result is ev
        assert len(ev.metrics) == 2

    def test_evaluate_single_sample(self, perfect_sample: EvalSample) -> None:
        ev = Evaluator(metrics=[FaithfulnessMetric(), TokenF1Metric()])
        report = ev.evaluate([perfect_sample])
        assert len(report.sample_results) == 1
        sr = report.sample_results[0]
        assert "faithfulness" in sr.metrics
        assert "token_f1" in sr.metrics

    def test_evaluate_multiple_samples(
        self,
        perfect_sample: EvalSample,
        partial_sample: EvalSample,
        no_reference_sample: EvalSample,
    ) -> None:
        ev = Evaluator(metrics=[FaithfulnessMetric()])
        report = ev.evaluate([perfect_sample, partial_sample, no_reference_sample])
        assert len(report.sample_results) == 3

    def test_aggregate_scores(self, perfect_sample: EvalSample, partial_sample: EvalSample) -> None:
        ev = Evaluator(metrics=[FaithfulnessMetric()])
        report = ev.evaluate([perfect_sample, partial_sample])
        agg = report.aggregate
        assert "faithfulness" in agg
        assert 0.0 <= agg["faithfulness"] <= 1.0

    def test_empty_samples(self) -> None:
        ev = Evaluator()
        report = ev.evaluate([])
        assert report.aggregate == {}

    def test_evaluate_single_helper(self, perfect_sample: EvalSample) -> None:
        ev = Evaluator(metrics=[FaithfulnessMetric()])
        sr = ev.evaluate_single(perfect_sample)
        assert isinstance(sr, SampleResult)
        assert "faithfulness" in sr.metrics

    def test_sample_id_auto_assigned(self) -> None:
        sample = EvalSample(question="Q", contexts=[], answer="A")
        ev = Evaluator(metrics=[FaithfulnessMetric()])
        report = ev.evaluate([sample])
        assert report.sample_results[0].sample_id == "0"

    def test_sample_id_preserved(self, perfect_sample: EvalSample) -> None:
        ev = Evaluator(metrics=[FaithfulnessMetric()])
        report = ev.evaluate([perfect_sample])
        assert report.sample_results[0].sample_id == "perfect-001"

    def test_scores_property(self, perfect_sample: EvalSample) -> None:
        ev = Evaluator(metrics=[FaithfulnessMetric(), TokenF1Metric()])
        report = ev.evaluate([perfect_sample])
        scores = report.sample_results[0].scores
        assert isinstance(scores, dict)
        assert all(isinstance(v, float) for v in scores.values())


# ---------------------------------------------------------------------------
# EvalReport tests
# ---------------------------------------------------------------------------


class TestEvalReport:
    def test_summary_output(self, perfect_sample: EvalSample) -> None:
        ev = Evaluator(metrics=[FaithfulnessMetric()])
        report = ev.evaluate([perfect_sample])
        summary = report.summary()
        assert "LexiSearch Evaluation Report" in summary
        assert "faithfulness" in summary
        assert "Samples evaluated: 1" in summary

    def test_aggregate_empty(self) -> None:
        report = EvalReport()
        assert report.aggregate == {}

    def test_aggregate_multiple_metrics(self, perfect_sample: EvalSample) -> None:
        ev = Evaluator(metrics=[FaithfulnessMetric(), TokenF1Metric()])
        report = ev.evaluate([perfect_sample])
        agg = report.aggregate
        assert len(agg) == 2
