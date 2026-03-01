"""Orchestrates RAG evaluation across multiple metrics and samples."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lexisearch.evaluation.base import BaseMetric, EvalReport, EvalSample, SampleResult
from lexisearch.evaluation.metrics import (
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ExactMatchMetric,
    FaithfulnessMetric,
    TokenF1Metric,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_METRICS: list[BaseMetric] = [
    FaithfulnessMetric(),
    ContextPrecisionMetric(),
    ContextRecallMetric(),
    AnswerRelevanceMetric(),
    TokenF1Metric(),
    ExactMatchMetric(),
]


class Evaluator:
    """Orchestrates RAG evaluation across multiple metrics and samples.

    By default the evaluator runs all six built-in metrics:
    faithfulness, context_precision, context_recall, answer_relevance,
    token_f1, and exact_match.

    Example:
        >>> from lexisearch.evaluation import Evaluator, EvalSample
        >>> evaluator = Evaluator()
        >>> sample = EvalSample(
        ...     question="What is RAG?",
        ...     contexts=["RAG stands for Retrieval-Augmented Generation."],
        ...     answer="RAG is Retrieval-Augmented Generation.",
        ...     reference="RAG stands for Retrieval-Augmented Generation.",
        ... )
        >>> report = evaluator.evaluate([sample])
        >>> print(report.summary())
    """

    def __init__(
        self,
        metrics: Sequence[BaseMetric] | None = None,
    ) -> None:
        """Initialise the Evaluator.

        Args:
            metrics: Optional list of :class:`BaseMetric` instances to use.
                If ``None``, all six default metrics are used.
        """
        self._metrics: list[BaseMetric] = (
            list(metrics) if metrics is not None else list(_DEFAULT_METRICS)
        )

    @property
    def metrics(self) -> list[BaseMetric]:
        """Return the list of metrics registered with this evaluator."""
        return list(self._metrics)

    def add_metric(self, metric: BaseMetric) -> Evaluator:
        """Register an additional metric and return ``self`` (fluent API).

        Args:
            metric: A :class:`BaseMetric` instance to add.

        Returns:
            This Evaluator (for method chaining).
        """
        self._metrics.append(metric)
        return self

    def evaluate(self, samples: Sequence[EvalSample]) -> EvalReport:
        """Evaluate a list of samples against all registered metrics.

        Args:
            samples: Sequence of :class:`EvalSample` instances to evaluate.

        Returns:
            :class:`EvalReport` containing per-sample and aggregate results.
        """
        report = EvalReport()

        for i, sample in enumerate(samples):
            sample_id = sample.sample_id or str(i)
            sr = SampleResult(sample_id=sample_id, question=sample.question)

            for metric in self._metrics:
                result = metric.score(sample)
                sr.metrics[result.metric_name] = result

            report.sample_results.append(sr)

        return report

    def evaluate_single(self, sample: EvalSample) -> SampleResult:
        """Evaluate a single sample against all registered metrics.

        Args:
            sample: A single :class:`EvalSample` to evaluate.

        Returns:
            :class:`SampleResult` with scores from each metric.
        """
        return self.evaluate([sample]).sample_results[0]
