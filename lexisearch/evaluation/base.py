"""Base classes for LexiSearch RAG evaluation."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalSample:
    """A single evaluation sample with question, context, answer, and optional reference.

    Attributes:
        question: The input question or query.
        contexts: List of retrieved context strings (chunks).
        answer: The generated answer from the RAG pipeline.
        reference: Optional ground-truth reference answer.
        sample_id: Optional identifier for this sample.
        metadata: Optional dict of extra metadata.
    """

    question: str
    contexts: list[str]
    answer: str
    reference: str | None = None
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Result for a single metric evaluated on one sample.

    Attributes:
        metric_name: Name of the metric that produced this result.
        score: Score in the range [0.0, 1.0].
        details: Optional dict with extra diagnostic information.
    """

    metric_name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """Evaluation results for a single EvalSample.

    Attributes:
        sample_id: Identifier for this sample (may be auto-assigned).
        question: The original question.
        metrics: Mapping of metric name to MetricResult.
    """

    sample_id: str | None
    question: str
    metrics: dict[str, MetricResult] = field(default_factory=dict)

    @property
    def scores(self) -> dict[str, float]:
        """Return a mapping of metric name to numeric score."""
        return {name: r.score for name, r in self.metrics.items()}


@dataclass
class EvalReport:
    """Aggregate evaluation report across all samples.

    Attributes:
        sample_results: List of per-sample evaluation results.
    """

    sample_results: list[SampleResult] = field(default_factory=list)

    @property
    def aggregate(self) -> dict[str, float]:
        """Return average score per metric across all samples."""
        if not self.sample_results:
            return {}

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}

        for sr in self.sample_results:
            for name, score in sr.scores.items():
                totals[name] = totals.get(name, 0.0) + score
                counts[name] = counts.get(name, 0) + 1

        return {name: totals[name] / counts[name] for name in totals}

    def summary(self) -> str:
        """Return a human-readable summary of the evaluation report."""
        lines = ["=== LexiSearch Evaluation Report ==="]
        lines.append(f"Samples evaluated: {len(self.sample_results)}")
        lines.append("")
        lines.append("Aggregate Scores:")
        for metric, score in sorted(self.aggregate.items()):
            lines.append(f"  {metric}: {score:.4f}")
        return "\n".join(lines)


class BaseMetric(ABC):
    """Abstract base class for RAG evaluation metrics.

    Subclasses must implement ``name`` and ``score``.
    Helper tokenisation utilities are provided as static methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric identifier (e.g. ``faithfulness``)."""
        ...

    @abstractmethod
    def score(self, sample: EvalSample) -> MetricResult:
        """Score a single evaluation sample.

        Args:
            sample: The EvalSample to evaluate.

        Returns:
            A MetricResult with a score in [0.0, 1.0].
        """
        ...

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase word tokens.

        Args:
            text: Input string.

        Returns:
            List of lowercase word tokens.
        """
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _token_set(text: str) -> set[str]:
        """Return the set of unique lowercase tokens from text.

        Args:
            text: Input string.

        Returns:
            Set of unique lowercase word tokens.
        """
        return set(re.findall(r"\b\w+\b", text.lower()))

    @staticmethod
    def _f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
        """Compute token-level F1 score between prediction and reference.

        Args:
            pred_tokens: Tokens from the predicted answer.
            ref_tokens: Tokens from the reference answer.

        Returns:
            F1 score in [0.0, 1.0].
        """
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)

        if not pred_set and not ref_set:
            return 1.0
        if not pred_set or not ref_set:
            return 0.0

        overlap = pred_set & ref_set
        precision = len(overlap) / len(pred_set)
        recall = len(overlap) / len(ref_set)

        if precision + recall == 0.0:
            return 0.0

        return 2.0 * precision * recall / (precision + recall)

    @staticmethod
    def _jaccard(set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard similarity between two token sets.

        Args:
            set_a: First token set.
            set_b: Second token set.

        Returns:
            Jaccard similarity in [0.0, 1.0].
        """
        if not set_a and not set_b:
            return 1.0
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)
