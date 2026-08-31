"""
Shared TraceRecord.to_dict()-shaped payload builder for collector tests.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from conntrail.contrast import ContrastSet
from conntrail.record import TraceRecord


def make_trace_payload(
    *,
    node_id: str = "router",
    timestamp: datetime | None = None,
    stability: str = "confident",
    status: str = "ok",
    entropy_score: float = 0.1,
    **overrides: Any,
) -> dict:
    """Build a valid TraceRecord.to_dict() payload for tests."""
    record = TraceRecord(
        trace_id=TraceRecord.make_id(),
        node_id=node_id,
        timestamp=timestamp or datetime.now(UTC),
        original_input="I need a refund",
        original_route="refund",
        entropy_score=entropy_score,
        stability=stability,
        attribution_dimension="urgency",
        plain_language_summary="The 'router' node routed to 'refund' with confident confidence.",
        raw_contrasts=ContrastSet(similar="a", neutral="b", opposite="c"),
        raw_outputs={"similar": "refund", "neutral": "refund", "opposite": "escalation"},
        counterfactual_route="escalation",
        status=status,
        error_type=None if status == "ok" else "ValueError",
        error_message=None if status == "ok" else "boom",
    )
    payload = record.to_dict()
    payload.update(overrides)
    return payload
