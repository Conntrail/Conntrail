"""
POST /v1/traces — collector ingest endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from conntrail_server.auth import require_api_key
from conntrail_server.classifier import classify_failure
from conntrail_server.models import TraceIngestResponse, TraceRecordModel

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/v1/traces", status_code=201, response_model=TraceIngestResponse)
async def ingest_trace(payload: TraceRecordModel, request: Request) -> TraceIngestResponse:
    record = payload.to_store_dict()
    # Server-computed, not client-trusted — overwrite whatever the client sent (if anything).
    record["failure_category"] = classify_failure(record)
    trace_id = request.app.state.store.insert(record)
    return TraceIngestResponse(trace_id=trace_id)
