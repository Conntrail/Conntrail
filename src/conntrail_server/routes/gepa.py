"""
POST /v1/gepa-attempts, GET /v1/gepa-attempts?run_id=... — persists and
exposes CPE-GEPA run results (Phase 5's G2 live run), so Phase 4's D2
before/after panel has something to read.

A PromptAttemptRecord (src/conntrail/gepa/schema.py) has no run_id of its
own — one live run produces many attempts, and D2 needs to group them —
so run_id is supplied by the caller (run_live.py) alongside the attempt
payload, not part of PromptAttemptRecord's own shape.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from conntrail_server.auth import require_api_key
from conntrail_server.models import TraceRecordModel

router = APIRouter(dependencies=[Depends(require_api_key)])


class GepaAttemptIn(BaseModel):
    run_id: str
    attempt_id: str
    prompt_candidate: str
    scalar_score: float | None = None
    traces: list[TraceRecordModel]
    # Optional attempt-level cost telemetry (stamped by the run harness when
    # the optimizing run captured cost; derivable from traces otherwise).
    token_usage: dict[str, int] | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None

    def to_store_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "prompt_candidate": self.prompt_candidate,
            "scalar_score": self.scalar_score,
            "traces": [t.model_dump() for t in self.traces],
            "token_usage": self.token_usage,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
        }


class GepaAttemptIngestResponse(BaseModel):
    attempt_id: str


class GepaAttemptOut(BaseModel):
    run_id: str
    attempt_id: str
    prompt_candidate: str
    scalar_score: float | None
    traces: list[TraceRecordModel]
    token_usage: dict[str, int] | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None


class GepaAttemptListResponse(BaseModel):
    run_id: str
    attempts: list[GepaAttemptOut]


@router.post("/v1/gepa-attempts", status_code=201, response_model=GepaAttemptIngestResponse)
async def ingest_gepa_attempt(
    payload: GepaAttemptIn, request: Request
) -> GepaAttemptIngestResponse:
    attempt_id = request.app.state.store.insert_gepa_attempt(payload.to_store_dict())
    return GepaAttemptIngestResponse(attempt_id=attempt_id)


@router.get("/v1/gepa-attempts", response_model=GepaAttemptListResponse)
async def list_gepa_attempts(
    request: Request, run_id: str = Query(...)
) -> GepaAttemptListResponse:
    rows = request.app.state.store.list_gepa_attempts(run_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"no attempts found for run_id={run_id!r}")
    return GepaAttemptListResponse(
        run_id=run_id,
        attempts=[GepaAttemptOut(**row) for row in rows],
    )
