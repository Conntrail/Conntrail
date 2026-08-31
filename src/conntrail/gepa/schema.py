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
