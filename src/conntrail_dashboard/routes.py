"""
Dashboard routes — trace list (filterable), per-trace detail, failure view.

Server-rendered Jinja2 + HTMX: filter/pagination requests carry the
HX-Request header and get back just the row partial to swap in, instead of
a full page reload.
"""
from __future__ import annotations

from collections import Counter
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

STABILITIES = ("confident", "boundary", "fragile")
FAILURE_CATEGORIES = ("exception", "retry_loop", "timeout", "malformed_output", "none")
_FAILURE_CATEGORIES_EXCLUDING_NONE = ("exception", "retry_loop", "timeout", "malformed_output")

# GET /v1/traces has no total-count facility (T3's list is page-only), so
# failure counts are approximated by fetching up to this many rows per
# category and counting what came back. Fine at this scale (sqlite, single
# collector); a real count endpoint would replace this if traffic grew.
_COUNT_SAMPLE_LIMIT = 1000


def _collector(request: Request):
    return request.app.state.collector


def _render(request: Request, full_template: str, partial_template: str, context: dict):
    templates = request.app.state.templates
    template_name = partial_template if "HX-Request" in request.headers else full_template
    return templates.TemplateResponse(request, template_name, context)


def _query_string(**filters: str | None) -> str:
    return urlencode({k: v for k, v in filters.items() if v})


@router.get("/", response_class=HTMLResponse)
async def trace_list(
    request: Request,
    node_id: str | None = None,
    stability: str | None = None,
    failure_category: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    data = await _collector(request).list_traces(
        node_id=node_id,
        stability=stability,
        failure_category=failure_category,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )
    filters = {
        "node_id": node_id,
        "stability": stability,
        "failure_category": failure_category,
        "since": since,
        "until": until,
    }
    context = {
        "traces": data["traces"],
        "limit": data["limit"],
        "offset": data["offset"],
        "filters": filters,
        "filters_qs": _query_string(**filters),
        "stabilities": STABILITIES,
        "failure_categories": FAILURE_CATEGORIES,
    }
    return _render(request, "trace_list.html", "_trace_rows.html", context)


@router.get("/traces/{trace_id}", response_class=HTMLResponse)
async def trace_detail(request: Request, trace_id: str):
    trace = await _collector(request).get_trace(trace_id)
    if trace is None:
        return request.app.state.templates.TemplateResponse(
            request,
            "not_found.html",
            {"trace_id": trace_id},
            status_code=404,
        )
    return request.app.state.templates.TemplateResponse(
        request, "trace_detail.html", {"trace": trace}
    )


@router.get("/failures", response_class=HTMLResponse)
async def failure_view(
    request: Request,
    failure_category: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    collector = _collector(request)

    counts: dict[str, int] = {}
    for category in _FAILURE_CATEGORIES_EXCLUDING_NONE:
        page = await collector.list_traces(
            failure_category=category, limit=_COUNT_SAMPLE_LIMIT, offset=0
        )
        counts[category] = len(page["traces"])

    traces: list = []
    if failure_category:
        page = await collector.list_traces(
            failure_category=failure_category, limit=limit, offset=offset
        )
        traces = page["traces"]

    context = {
        "traces": traces,
        "counts": counts,
        "failure_categories": _FAILURE_CATEGORIES_EXCLUDING_NONE,
        "selected_category": failure_category,
        "count_sample_limit": _COUNT_SAMPLE_LIMIT,
    }
    return _render(request, "failure_view.html", "_failure_rows.html", context)


def _attempt_summary(attempt: dict[str, Any]) -> dict[str, Any]:
    """Mirror PromptAttemptRecord's mean_entropy/fragile_count/boundary_count/
    dominant_attribution properties (src/conntrail/gepa/schema.py) — G3's API
    stores an attempt's raw traces, not these derived numbers, so the
    dashboard recomputes them the same way the dataclass would.
    """
    traces = attempt.get("traces") or []
    n = len(traces)
    fragile_count = sum(1 for t in traces if t.get("stability") == "fragile")
    boundary_count = sum(1 for t in traces if t.get("stability") == "boundary")
    mean_entropy = sum(t.get("entropy_score", 0.0) for t in traces) / n if n else None
    dominant_attribution = None
    if traces:
        counts = Counter(t.get("attribution_dimension") for t in traces)
        dominant_attribution = counts.most_common(1)[0][0]

    return {
        "attempt_id": attempt["attempt_id"],
        "prompt_candidate": attempt["prompt_candidate"],
        "scalar_score": attempt.get("scalar_score"),
        "num_traces": n,
        "mean_entropy": mean_entropy,
        "fragile_count": fragile_count,
        "boundary_count": boundary_count,
        "confident_count": n - fragile_count - boundary_count,
        "dominant_attribution": dominant_attribution,
    }


def _delta(last: dict[str, Any], first: dict[str, Any], key: str) -> float | None:
    a, b = last.get(key), first.get(key)
    return a - b if a is not None and b is not None else None


@router.get("/before-after", response_class=HTMLResponse)
async def before_after(request: Request, run_id: str | None = None):
    context: dict[str, Any] = {"run_id": run_id, "error": None, "first": None, "last": None}

    if run_id:
        data = await _collector(request).get_gepa_attempts(run_id)
        if data is None:
            context["error"] = f"No GEPA run found with run_id={run_id!r}."
        elif not data["attempts"]:
            context["error"] = "This run has no recorded attempts."
        else:
            attempts = data["attempts"]
            first = _attempt_summary(attempts[0])
            last = _attempt_summary(attempts[-1])
            context["first"] = first
            context["last"] = last
            context["num_attempts"] = len(attempts)
            context["deltas"] = {
                "mean_entropy": _delta(last, first, "mean_entropy"),
                "scalar_score": _delta(last, first, "scalar_score"),
                "fragile_count": _delta(last, first, "fragile_count"),
                "boundary_count": _delta(last, first, "boundary_count"),
                "confident_count": _delta(last, first, "confident_count"),
            }

    return request.app.state.templates.TemplateResponse(request, "before_after.html", context)
