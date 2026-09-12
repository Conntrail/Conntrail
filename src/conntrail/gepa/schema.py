from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from conntrail.record import TraceRecord


@dataclass
class PromptAttemptRecord:
    """Links a GEPA prompt candidate string to the Conntrail traces it produced."""

    attempt_id: str
    prompt_candidate: str
    traces: list[TraceRecord] = field(default_factory=list)
    scalar_score: float | None = None
    # Attempt-level cost telemetry. Authoritative when stamped by the run
    # harness (e.g. run_live.py stamps dspy's usage tracker totals — dspy LM
    # calls are invisible to the SDK's LangChain callback capture); the
    # derived properties below fall back to aggregating the embedded traces'
    # telemetry when these are None.
    token_usage: dict[str, int] | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None

    @property
    def mean_entropy(self) -> float | None:
        if not self.traces:
            return None
        return sum(t.entropy_score for t in self.traces) / len(self.traces)

    @property
    def fragile_count(self) -> int:
        return sum(1 for t in self.traces if t.stability == "fragile")

    @property
    def boundary_count(self) -> int:
        return sum(1 for t in self.traces if t.stability == "boundary")

    @property
    def dominant_attribution(self) -> str | None:
        """Most common attribution_dimension across traces."""
        if not self.traces:
            return None
        counts = Counter(t.attribution_dimension for t in self.traces)
        return counts.most_common(1)[0][0]

    # -- cost telemetry (attempt-level first, trace-derived fallback) --------

    @property
    def total_input_tokens(self) -> int | None:
        if self.token_usage is not None:
            return self.token_usage.get("input_tokens")
        values = [
            t.token_usage.get("input_tokens")
            for t in self.traces
            if isinstance(t.token_usage, dict)
        ]
        values = [v for v in values if v is not None]
        return sum(values) if values else None

    @property
    def total_output_tokens(self) -> int | None:
        if self.token_usage is not None:
            return self.token_usage.get("output_tokens")
        values = [
            t.token_usage.get("output_tokens")
            for t in self.traces
            if isinstance(t.token_usage, dict)
        ]
        values = [v for v in values if v is not None]
        return sum(values) if values else None

    @property
    def total_cost_usd(self) -> float | None:
        if self.cost_usd is not None:
            return self.cost_usd
        costs = [t.cost_usd for t in self.traces if t.cost_usd is not None]
        return round(sum(costs), 6) if costs else None

    @property
    def mean_latency_ms(self) -> float | None:
        if self.latency_ms is not None:
            return self.latency_ms
        latencies = [t.latency_ms for t in self.traces if t.latency_ms is not None]
        return sum(latencies) / len(latencies) if latencies else None
