"""
GET /v1/traces, GET /v1/traces/{trace_id} — collector query endpoints.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from conntrail_server.auth import require_api_key
from conntrail_server.models import TraceListResponse, TraceRecordModel

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/v1/traces", response_model=TraceListResponse)
async def list_traces(
    request: Request,
    node_id: str | None = None,
    stability: Literal["confident", "boundary", "fragile"] | None = None,
    status: Literal["ok", "error"] | None = None,
    failure_category: Literal["exception", "retry_loop", "timeout", "malformed_output", "none"]
    | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> TraceListResponse:
    rows = request.app.state.store.list(
        node_id=node_id,
        stability=stability,
        status=status,
        failure_category=failure_category,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )
    return TraceListResponse(
        traces=[TraceRecordModel(**row) for row in rows], limit=limit, offset=offset
    )


@router.get("/v1/traces/{trace_id}", response_model=TraceRecordModel)
async def get_trace(trace_id: str, request: Request) -> TraceRecordModel:
    row = request.app.state.store.get(trace_id)
    if row is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return TraceRecordModel(**row)
