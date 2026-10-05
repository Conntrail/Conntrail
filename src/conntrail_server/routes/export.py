"""
GET /v1/export — a bundled snapshot for report generation.

Returns the cost summary plus the per-trace records it was computed from, in
one round trip, with provenance (generated_at, filters, price disclaimer). The
report generator (conntrail_reports) consumes this shape; the same shape can
be frozen to a JSON file for offline reports.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from conntrail_server.auth import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])

MAX_TRACES = 5000


@router.get("/v1/export")
async def export_bundle(
    request: Request,
    node_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = Query(1000, ge=1, le=MAX_TRACES),
) -> dict[str, Any]:
    store = request.app.state.store
    summary = store.cost_summary(limit=limit)
    traces = store.list(
        node_id=node_id, since=since, until=until, limit=limit, offset=0
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "filters": {"node_id": node_id, "since": since, "until": until, "limit": limit},
        "cost_summary": {
            "nodes": summary["nodes"],
            "shared_prompt_blocks": summary["shared_prompt_blocks"],
            "scanned_traces": sum(node["trace_count"] for node in summary["nodes"]),
        },
        "traces": traces,
    }
