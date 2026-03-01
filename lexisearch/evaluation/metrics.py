"""Concrete RAG evaluation metric implementations."""

from __future__ import annotations

import re

from lexisearch.evaluation.base import BaseMetric, EvalSample, MetricResult


class FaithfulnessMetric(BaseMetric):
    """Measures whether the answer is supported by the retrieved contexts.

    Approximates faithfulness as the fraction of answer tokens that appear
    in the combined retrieved context. A score of 1.0 means every answer
    token is present in the context.
    """

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "faithfulness"

    def score(self, sample: EvalSample) -> MetricResult:
        """Score the faithfulness of the answer relative to the context.

        Args:
            sample: The EvalSample containing answer and contexts.

        Returns:
            MetricResult with faithfulness score in [0.0, 1.0].
        """
        combined_context = " ".join(sample.contexts)
        context_tokens = self._token_set(combined_context)
        answer_tokens = self._tokenize(sample.answer)

        if not answer_tokens:
            return MetricResult(metric_name=self.name, score=1.0)

        supported = sum(1 for t in answer_tokens if t in context_tokens)
        faithfulness_score = supported / len(answer_tokens)

        return MetricResult(
            metric_name=self.name,
            score=faithfulness_score,
            details={
                "answer_tokens": len(answer_tokens),
                "supported_tokens": supported,
                "context_length": len(combined_context),
            },
        )


class ContextPrecisionMetric(BaseMetric):
    """Measures what fraction of retrieved contexts are relevant to the question.

    A context is considered relevant if its Jaccard similarity with the
    combined question and answer exceeds ``relevance_threshold``.
    """

    def __init__(self, relevance_threshold: float = 0.05) -> None:
        """Initialise with a relevance threshold.

        Args:
            relevance_threshold: Minimum Jaccard similarity for a context
                to be counted as relevant.  Defaults to 0.05.
        """
        self._threshold = relevance_threshold

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "context_precision"

    def score(self, sample: EvalSample) -> MetricResult:
        """Score the precision of the retrieved context set.

        Args:
            sample: The EvalSample to evaluate.

        Returns:
            MetricResult with context precision in [0.0, 1.0].
        """
        if not sample.contexts:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                details={"total_contexts": 0, "relevant_contexts": 0},
            )

        query_tokens = self._token_set(sample.question + " " + sample.answer)

        relevance_scores = [
            self._jaccard(query_tokens, self._token_set(ctx)) for ctx in sample.contexts
        ]

        relevant_count = sum(1 for s in relevance_scores if s >= self._threshold)
        precision = relevant_count / len(sample.contexts)

        return MetricResult(
            metric_name=self.name,
            score=precision,
            details={
                "total_contexts": len(sample.contexts),
                "relevant_contexts": relevant_count,
                "relevance_scores": relevance_scores,
            },
        )


class ContextRecallMetric(BaseMetric):
    """Measures how much of the reference answer is covered by the contexts.

    Uses token-overlap between the reference (or the answer when no reference
    is provided) and the combined retrieved context.
    """

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "context_recall"

    def score(self, sample: EvalSample) -> MetricResult:
        """Score context recall for a sample.

        Args:
            sample: The EvalSample to evaluate.

        Returns:
            MetricResult with context recall in [0.0, 1.0].
        """
        reference = sample.reference if sample.reference is not None else sample.answer
        combined_context = " ".join(sample.contexts)

        ref_tokens = self._tokenize(reference)
        context_tokens = self._token_set(combined_context)

        if not ref_tokens:
            return MetricResult(metric_name=self.name, score=1.0)

        recalled = sum(1 for t in ref_tokens if t in context_tokens)
        recall_score = recalled / len(ref_tokens)

        return MetricResult(
            metric_name=self.name,
            score=recall_score,
            details={
                "reference_tokens": len(ref_tokens),
                "recalled_tokens": recalled,
                "has_reference": sample.reference is not None,
            },
        )


class AnswerRelevanceMetric(BaseMetric):
    """Measures whether the generated answer is relevant to the question.

    Uses Jaccard similarity between the question token set and the answer
    token set as a lightweight proxy for answer relevance.
    """

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "answer_relevance"

    def score(self, sample: EvalSample) -> MetricResult:
        """Score the relevance of the answer to the question.

        Args:
            sample: The EvalSample to evaluate.

        Returns:
            MetricResult with answer relevance in [0.0, 1.0].
        """
        question_tokens = self._token_set(sample.question)
        answer_tokens = self._token_set(sample.answer)

        relevance = self._jaccard(question_tokens, answer_tokens)

        return MetricResult(
            metric_name=self.name,
            score=relevance,
            details={
                "question_tokens": len(question_tokens),
                "answer_tokens": len(answer_tokens),
                "shared_tokens": len(question_tokens & answer_tokens),
            },
        )


class ExactMatchMetric(BaseMetric):
    """Measures whether the normalised answer exactly matches the reference.

    Normalisation lowercases the text, strips punctuation, and collapses
    whitespace before comparison.  Returns 0.0 when no reference is provided.
    """

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "exact_match"

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalise text for comparison.

        Args:
            text: Raw text string.

        Returns:
            Lowercase, punctuation-free, whitespace-collapsed string.
        """
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        return " ".join(text.split())

    def score(self, sample: EvalSample) -> MetricResult:
        """Score whether the answer exactly matches the reference.

        Args:
            sample: The EvalSample to evaluate.

        Returns:
            MetricResult with score 1.0 for exact match, 0.0 otherwise.
        """
        if sample.reference is None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                details={"error": "no reference provided"},
            )

        normalized_answer = self._normalize(sample.answer)
        normalized_reference = self._normalize(sample.reference)

        match_score = 1.0 if normalized_answer == normalized_reference else 0.0

        return MetricResult(
            metric_name=self.name,
            score=match_score,
            details={
                "normalized_answer": normalized_answer,
                "normalized_reference": normalized_reference,
            },
        )


class TokenF1Metric(BaseMetric):
    """Token-level F1 score between the generated answer and the reference.

    Returns 0.0 when no reference is provided.
    """

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "token_f1"

    def score(self, sample: EvalSample) -> MetricResult:
        """Compute token-level F1 between answer and reference.

        Args:
            sample: The EvalSample to evaluate.

        Returns:
            MetricResult with token F1 in [0.0, 1.0].
        """
        if sample.reference is None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                details={"error": "no reference provided"},
            )

        answer_tokens = self._tokenize(sample.answer)
        reference_tokens = self._tokenize(sample.reference)

        f1 = self._f1(answer_tokens, reference_tokens)

        return MetricResult(
            metric_name=self.name,
            score=f1,
            details={
                "answer_tokens": len(answer_tokens),
                "reference_tokens": len(reference_tokens),
            },
        )
