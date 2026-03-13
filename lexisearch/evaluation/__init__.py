"""Evaluation module for LexiSearch RAG pipelines.

Provides deterministic, dependency-free metrics and an :class:`Evaluator`
class for measuring the quality of retrieval-augmented generation systems.

Built-in metrics (all return scores in [0.0, 1.0]):

* **faithfulness** -- Is the answer supported by the retrieved context?
* **context_precision** -- Are the retrieved contexts relevant to the query?
* **context_recall** -- Does the context cover the reference answer?
* **answer_relevance** -- Is the answer relevant to the question?
* **exact_match** -- Does the answer exactly match the reference?
* **token_f1** -- Token-level F1 between answer and reference.

Example::

    from lexisearch.evaluation import Evaluator, EvalSample

    evaluator = Evaluator()
    sample = EvalSample(
        question="What is RAG?",
        contexts=["RAG stands for Retrieval-Augmented Generation."],
        answer="RAG is Retrieval-Augmented Generation.",
        reference="RAG stands for Retrieval-Augmented Generation.",
    )
    report = evaluator.evaluate([sample])
    print(report.summary())
"""

from __future__ import annotations

from lexisearch.evaluation.base import (
    BaseMetric,
    EvalReport,
    EvalSample,
    MetricResult,
    SampleResult,
)
from lexisearch.evaluation.evaluator import Evaluator
from lexisearch.evaluation.ir_metrics import (
    RetrievalEvalReport,
    average_precision,
    compute_ir_metrics,
    mean_average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from lexisearch.evaluation.metrics import (
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ExactMatchMetric,
    FaithfulnessMetric,
    TokenF1Metric,
)

__all__ = [
    "AnswerRelevanceMetric",
    "BaseMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "EvalReport",
    "EvalSample",
    "Evaluator",
    "ExactMatchMetric",
    "FaithfulnessMetric",
    "MetricResult",
    "RetrievalEvalReport",
    "SampleResult",
    "TokenF1Metric",
    "average_precision",
    "compute_ir_metrics",
    "mean_average_precision",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
