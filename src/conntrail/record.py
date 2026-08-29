"""
TraceRecord — holds the full analysis output for one node call.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from conntrail.contrast import ContrastSet


@dataclass
class TraceRecord:
    """
    Complete trace output for a single node invocation.

    Produced by DivergenceAnalyser and consumed by exporters.
    Attached to agent state under the key ``__conntrail_traces__``.
    """

    trace_id: str
    node_id: str
    timestamp: datetime
    original_input: str
    original_route: str
    entropy_score: float
    stability: Literal["confident", "boundary", "fragile"]
    attribution_dimension: str
    plain_language_summary: str
    raw_contrasts: ContrastSet
    raw_outputs: dict[str, Any]
    counterfactual_route: str | None = None
    status: Literal["ok", "error"] = "ok"
    error_type: str | None = None
    error_message: str | None = None

    @classmethod
    def make_id(cls) -> str:
        return str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "trace_id": self.trace_id,
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "original_input": self.original_input,
            "original_route": self.original_route,
            "entropy_score": self.entropy_score,
            "stability": self.stability,
            "attribution_dimension": self.attribution_dimension,
            "plain_language_summary": self.plain_language_summary,
            "raw_contrasts": {
                "similar": self.raw_contrasts.similar,
                "neutral": self.raw_contrasts.neutral,
                "opposite": self.raw_contrasts.opposite,
            },
            "raw_outputs": self.raw_outputs,
            "counterfactual_route": self.counterfactual_route,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceRecord:
        """Reconstruct a TraceRecord from a serialised dictionary."""
        from conntrail.contrast import ContrastSet

        return cls(
            trace_id=data["trace_id"],
            node_id=data["node_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            original_input=data["original_input"],
            original_route=data["original_route"],
            entropy_score=data["entropy_score"],
            stability=data["stability"],
            attribution_dimension=data["attribution_dimension"],
            plain_language_summary=data["plain_language_summary"],
            raw_contrasts=ContrastSet(**data["raw_contrasts"]),
            raw_outputs=data["raw_outputs"],
            counterfactual_route=data.get("counterfactual_route"),
            status=data.get("status", "ok"),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
        )

    @staticmethod
    def stability_label(entropy: float) -> Literal["confident", "boundary", "fragile"]:
        """Map an entropy score to a stability label."""
        if entropy < 0.25:
            return "confident"
        elif entropy <= 0.60:
            return "boundary"
        return "fragile"

    @staticmethod
    def build_summary(
        node_id: str,
        route: str,
        stability: str,
        entropy: float,
        attribution: str,
        counterfactual: str | None,
    ) -> str:
        """Generate the plain-language summary string."""
        cf_part = (
            f" If that dimension were removed, the agent would likely have taken "
            f"the '{counterfactual}' path instead."
            if counterfactual
            else ""
        )
        return (
            f"The '{node_id}' node routed to '{route}' with {stability} confidence "
            f"(entropy: {entropy:.2f}). The decision appears to have been driven by "
            f"{attribution} in the input.{cf_part}"
        )
