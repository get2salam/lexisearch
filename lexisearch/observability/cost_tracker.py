"""Token usage and cost tracking for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Per-model pricing table (USD per 1 000 tokens)
# ---------------------------------------------------------------------------


#: Default pricing table (USD per 1 000 tokens, input/output).
#: Override by passing ``pricing`` to :class:`CostTracker`.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.001, 0.002),
    # Anthropic
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-haiku-20241022": (0.0008, 0.004),
    "claude-3-opus-20240229": (0.015, 0.075),
    # Google
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    # Embedding models (output price unused)
    "text-embedding-3-small": (0.00002, 0.0),
    "text-embedding-3-large": (0.00013, 0.0),
    "text-embedding-ada-002": (0.0001, 0.0),
}


# ---------------------------------------------------------------------------
# Usage record
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Token consumption for a single LLM call.

    Attributes:
        model: Model identifier used for cost lookup.
        prompt_tokens: Tokens in the prompt / input.
        completion_tokens: Tokens in the completion / output.
        estimated_cost_usd: Estimated cost in USD (may be 0 if model unknown).
        metadata: Optional extra context (request ID, operation name, …).
    """

    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Tracks token usage and estimated cost across multiple LLM calls.

    Example:
        >>> tracker = CostTracker()
        >>> tracker.record("gpt-4o-mini", prompt_tokens=200, completion_tokens=50)
        >>> print(tracker.total_cost_usd)
        4.5e-05

    Attributes:
        pricing: Per-model pricing table ``{model: (input_per_1k, output_per_1k)}``.
    """

    def __init__(
        self,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialise the cost tracker.

        Args:
            pricing: Optional custom pricing table.  Defaults to
                :data:`DEFAULT_PRICING`.
        """
        self.pricing: dict[str, tuple[float, float]] = pricing or dict(DEFAULT_PRICING)
        self._records: list[TokenUsage] = []

    def estimate_cost(
        self,
        model: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Return estimated USD cost for a single call without recording it.

        Args:
            model: Model identifier.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.

        Returns:
            Estimated cost in USD (0.0 if model not in pricing table).
        """
        if model not in self.pricing:
            return 0.0
        input_rate, output_rate = self.pricing[model]
        return (prompt_tokens / 1000 * input_rate) + (completion_tokens / 1000 * output_rate)

    def record(
        self,
        model: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        **metadata: Any,
    ) -> TokenUsage:
        """Record token usage for one LLM call.

        Args:
            model: Model identifier used for this call.
            prompt_tokens: Prompt token count.
            completion_tokens: Completion token count.
            **metadata: Extra key-value context (e.g. ``operation="query"``).

        Returns:
            The created :class:`TokenUsage` record.
        """
        cost = self.estimate_cost(
            model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        usage = TokenUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=cost,
            metadata=dict(metadata),
        )
        self._records.append(usage)
        return usage

    @property
    def records(self) -> list[TokenUsage]:
        """All recorded :class:`TokenUsage` objects (oldest first)."""
        return list(self._records)

    @property
    def total_prompt_tokens(self) -> int:
        """Sum of all prompt tokens across recorded calls."""
        return sum(r.prompt_tokens for r in self._records)

    @property
    def total_completion_tokens(self) -> int:
        """Sum of all completion tokens across recorded calls."""
        return sum(r.completion_tokens for r in self._records)

    @property
    def total_tokens(self) -> int:
        """Sum of all tokens (prompt + completion)."""
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_cost_usd(self) -> float:
        """Total estimated cost in USD."""
        return sum(r.estimated_cost_usd for r in self._records)

    def by_model(self) -> dict[str, dict[str, Any]]:
        """Aggregate usage statistics grouped by model.

        Returns:
            Dictionary mapping model name to aggregated stats.
        """
        result: dict[str, dict[str, Any]] = {}
        for record in self._records:
            if record.model not in result:
                result[record.model] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                }
            entry = result[record.model]
            entry["calls"] += 1
            entry["prompt_tokens"] += record.prompt_tokens
            entry["completion_tokens"] += record.completion_tokens
            entry["total_tokens"] += record.total_tokens
            entry["estimated_cost_usd"] += record.estimated_cost_usd
        return result

    def summary(self) -> dict[str, Any]:
        """Return a top-level usage summary.

        Returns:
            Dictionary with totals and per-model breakdown.
        """
        return {
            "total_calls": len(self._records),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "by_model": self.by_model(),
        }

    def reset(self) -> None:
        """Clear all recorded usage data."""
        self._records.clear()
