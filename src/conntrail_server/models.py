"""
Pydantic models for the collector API — the wire contract between the SDK,
the collector, and (later) the dashboard. Mirrors
conntrail.record.TraceRecord.to_dict()'s shape exactly.

TraceRecordModel is used both as the ingest request body (T2) and the query
response body (T3) — the wire shape is defined in exactly one place.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


class ContrastSetModel(BaseModel):
    similar: str
    neutral: str
    opposite: str


class TraceRecordModel(BaseModel):
    """TraceRecord.to_dict() shape."""

    model_config = ConfigDict(extra="ignore")  # forward-compatible: unknown fields don't error

    trace_id: str
    node_id: str
    timestamp: str
    original_input: str
    original_route: str
    entropy_score: float
    stability: Literal["confident", "boundary", "fragile"]
    attribution_dimension: str
    plain_language_summary: str
    raw_contrasts: ContrastSetModel
    raw_outputs: dict[str, Any]
    counterfactual_route: str | None = None
    status: Literal["ok", "error"] = "ok"
    error_type: str | None = None
    error_message: str | None = None
    # --- cost telemetry (optional; None on legacy/uncaptured traces) ---
    token_usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    analysis_overhead: dict[str, Any] | None = None
    cost_findings: list[dict[str, Any]] | None = None
    # Server-computed (C2's classifier) — ignored on ingest input, populated on output.
    failure_category: Literal["exception", "retry_loop", "timeout", "malformed_output", "none"] | None = None

    @field_validator("timestamp")
    @classmethod
    def _validate_iso_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"invalid ISO 8601 timestamp: {v!r}") from exc
        return v

    def to_store_dict(self) -> dict[str, Any]:
        """Serialise to the plain-dict shape TraceStore.insert() expects."""
        return self.model_dump()


class TraceIngestResponse(BaseModel):
    trace_id: str


class TraceListResponse(BaseModel):
    traces: list[TraceRecordModel]
    limit: int
    offset: int


class CostSummaryResponse(BaseModel):
    """GET /v1/cost-summary — aggregated cost telemetry per node."""

    nodes: list[dict[str, Any]]
    shared_prompt_blocks: list[dict[str, Any]]
    scanned_traces: int
