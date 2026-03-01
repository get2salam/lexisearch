"""Evaluation routes — run RAG quality metrics over a batch of samples."""

from typing import Any


def _make_evaluate_router() -> Any:
    """Build and return the evaluate router (requires FastAPI)."""
    try:
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI and pydantic are required for the API server: pip install fastapi"
        ) from exc

    from lexisearch.evaluation import EvalSample, Evaluator
    from lexisearch.evaluation.metrics import (
        AnswerRelevanceMetric,
        ContextPrecisionMetric,
        ContextRecallMetric,
        ExactMatchMetric,
        FaithfulnessMetric,
        TokenF1Metric,
    )

    _all_metrics = {
        "faithfulness": FaithfulnessMetric,
        "context_precision": ContextPrecisionMetric,
        "context_recall": ContextRecallMetric,
        "answer_relevance": AnswerRelevanceMetric,
        "exact_match": ExactMatchMetric,
        "token_f1": TokenF1Metric,
    }

    class SampleBody(BaseModel):
        question: str
        contexts: list[str]
        answer: str
        reference: str = ""

    class EvalBody(BaseModel):
        samples: list[SampleBody] = Field(..., min_length=1)
        metrics: list[str] = Field(
            default_factory=list,
            description="Metric names to compute.  Empty = all.",
        )

    class MetricScore(BaseModel):
        metric: str
        score: float
        passed: bool

    class SampleResult(BaseModel):
        question: str
        metric_scores: list[MetricScore]
        overall: float

    class EvalResult(BaseModel):
        num_samples: int
        aggregate: dict[str, float]
        samples: list[SampleResult] = []
        status: str = "ok"
        message: str = ""

    router = APIRouter(prefix="/evaluate", tags=["evaluate"])

    @router.post("", response_model=EvalResult)
    async def evaluate(body: EvalBody) -> EvalResult:  # type: ignore[misc]
        """Evaluate a batch of RAG samples using built-in metrics."""
        # Resolve requested metrics
        requested = [m.lower() for m in body.metrics] if body.metrics else list(_all_metrics)
        unknown = [m for m in requested if m not in _all_metrics]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown metrics: {unknown}. Available: {list(_all_metrics)}",
            )

        metrics = [_all_metrics[m]() for m in requested]
        evaluator = Evaluator(metrics=metrics)

        samples = [
            EvalSample(
                question=s.question,
                contexts=s.contexts,
                answer=s.answer,
                reference=s.reference,
            )
            for s in body.samples
        ]

        try:
            report = evaluator.evaluate(samples)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Build per-sample results
        sample_results: list[SampleResult] = []
        for sr in report.sample_results:
            # sr.metrics is dict[name, MetricResult]; sr.scores is dict[name, float]
            mscores = [
                MetricScore(
                    metric=name,
                    score=float(score),
                    passed=float(score) >= 0.5,
                )
                for name, score in sr.scores.items()
            ]
            overall = sum(ms.score for ms in mscores) / len(mscores) if mscores else 0.0
            sample_results.append(
                SampleResult(
                    question=sr.question,
                    metric_scores=mscores,
                    overall=round(overall, 4),
                )
            )

        return EvalResult(
            num_samples=len(samples),
            aggregate={k: round(v, 4) for k, v in report.aggregate.items()},
            samples=sample_results,
            status="ok",
        )

    @router.get("/metrics", response_model=None)
    async def list_metrics() -> dict[str, Any]:  # type: ignore[misc]
        """List all available evaluation metric names."""
        return {"metrics": list(_all_metrics.keys())}

    return router
