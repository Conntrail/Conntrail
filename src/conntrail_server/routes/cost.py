"""
GET /v1/cost-summary — per-node aggregated cost telemetry.

Powers the dashboard's cost view: token totals, cache hit ratio, cost,
latency, and cost-warning counts per node, plus cross-node shared
instruction blocks (repeated instructions → shared cached-prefix candidates).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from conntrail_server.auth import require_api_key
from conntrail_server.models import CostSummaryResponse

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/v1/cost-summary", response_model=CostSummaryResponse)
async def cost_summary(
    request: Request,
    limit: int = Query(1000, ge=1, le=10_000),
) -> CostSummaryResponse:
    summary = request.app.state.store.cost_summary(limit=limit)
    return CostSummaryResponse(
        nodes=summary["nodes"],
        shared_prompt_blocks=summary["shared_prompt_blocks"],
        scanned_traces=sum(node["trace_count"] for node in summary["nodes"]),
    )
