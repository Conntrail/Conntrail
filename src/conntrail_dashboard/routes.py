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

    Cost stats prefer the attempt-level fields stamped at end_attempt (when
    the optimizing run captured cost); otherwise they aggregate the embedded
    traces' cost telemetry.
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

    attempt_usage = attempt.get("token_usage")
    if isinstance(attempt_usage, dict):
        input_tokens = attempt_usage.get("input_tokens") or 0
        output_tokens = attempt_usage.get("output_tokens") or 0
    else:
        input_tokens = sum(
            (t.get("token_usage") or {}).get("input_tokens") or 0
            for t in traces
            if isinstance(t.get("token_usage"), dict)
        )
        output_tokens = sum(
            (t.get("token_usage") or {}).get("output_tokens") or 0
            for t in traces
            if isinstance(t.get("token_usage"), dict)
        )

    cost_usd = attempt.get("cost_usd")
    if cost_usd is None:
        costs = [t.get("cost_usd") for t in traces if t.get("cost_usd") is not None]
        cost_usd = sum(costs) if costs else None

    latency_ms = attempt.get("latency_ms")
    if latency_ms is None:
        latencies = [t.get("latency_ms") for t in traces if t.get("latency_ms") is not None]
        latency_ms = sum(latencies) / len(latencies) if latencies else None

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
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_cost_usd": cost_usd,
        "mean_latency_ms": latency_ms,
    }


def _delta(last: dict[str, Any], first: dict[str, Any], key: str) -> float | None:
    a, b = last.get(key), first.get(key)
    return a - b if a is not None and b is not None else None


def _group_candidates(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group attempts by exact prompt text, preserving first-seen order.

    GEPA evaluates each prompt candidate several times (minibatches, full
    re-evals), so a run's attempts are samples of a handful of *candidates* —
    comparing two individual attempts says little about the optimization.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        groups.setdefault(attempt.get("prompt_candidate", ""), []).append(attempt)
    return [{"prompt_candidate": p, "attempts": a} for p, a in groups.items()]


def _candidate_summary(group: dict[str, Any]) -> dict[str, Any]:
    """Aggregate a prompt candidate's attempts into one comparable summary."""
    attempts = group["attempts"]
    traces = [t for a in attempts for t in (a.get("traces") or [])]
    n = len(traces)
    fragile_count = sum(1 for t in traces if t.get("stability") == "fragile")
    boundary_count = sum(1 for t in traces if t.get("stability") == "boundary")
    mean_entropy = sum(t.get("entropy_score", 0.0) for t in traces) / n if n else None
    dominant_attribution = None
    if traces:
        counts = Counter(t.get("attribution_dimension") for t in traces)
        dominant_attribution = counts.most_common(1)[0][0]

    scores = [a["scalar_score"] for a in attempts if a.get("scalar_score") is not None]
    per_attempt = [_attempt_summary(a) for a in attempts]
    total_input_tokens = sum(s["total_input_tokens"] for s in per_attempt)
    total_output_tokens = sum(s["total_output_tokens"] for s in per_attempt)
    costs = [s["total_cost_usd"] for s in per_attempt if s["total_cost_usd"] is not None]
    total_cost_usd = round(sum(costs), 6) if costs else None
    latencies = [s["mean_latency_ms"] for s in per_attempt if s["mean_latency_ms"] is not None]

    # Per-attempt means, over attempts that actually carry telemetry: candidates
    # are sampled a different number of times, so *totals* would make a
    # rarely-sampled candidate look artificially cheap.
    tokenised = [
        s for s in per_attempt if (s["total_input_tokens"] or s["total_output_tokens"])
    ]
    mean_input_tokens = sum(s["total_input_tokens"] for s in tokenised) / len(tokenised) if tokenised else 0
    mean_output_tokens = sum(s["total_output_tokens"] for s in tokenised) / len(tokenised) if tokenised else 0
    mean_cost_usd = round(sum(costs) / len(costs), 6) if costs else None

    return {
        "prompt_candidate": group["prompt_candidate"],
        "num_attempts": len(attempts),
        "num_traces": n,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "best_score": max(scores) if scores else None,
        "mean_entropy": mean_entropy,
        "fragile_count": fragile_count,
        "boundary_count": boundary_count,
        "confident_count": n - fragile_count - boundary_count,
        "dominant_attribution": dominant_attribution,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": total_cost_usd,
        "mean_input_tokens": mean_input_tokens,
        "mean_output_tokens": mean_output_tokens,
        "mean_cost_usd": mean_cost_usd,
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "is_seed": False,
        "is_best": False,
    }


def _select_best(candidates: list[dict[str, Any]]) -> int:
    """Index of the best candidate: highest mean task score, cheaper on ties.

    Falls back to the last candidate when no attempt carries a score (e.g. a
    run with no task metric) — that is what the optimizer ended on.
    """
    scored = [(i, c) for i, c in enumerate(candidates) if c["mean_score"] is not None]
    if not scored:
        return len(candidates) - 1
    best_index, _ = max(
        scored,
        key=lambda ic: (
            ic[1]["mean_score"],
            -(ic[1]["total_input_tokens"] + ic[1]["total_output_tokens"]),
        ),
    )
    return best_index


@router.get("/before-after", response_class=HTMLResponse)
async def before_after(request: Request, run_id: str | None = None):
    context: dict[str, Any] = {"run_id": run_id, "error": None, "seed": None, "best": None}

    if run_id:
        data = await _collector(request).get_gepa_attempts(run_id)
        if data is None:
            context["error"] = f"No GEPA run found with run_id={run_id!r}."
        elif not data["attempts"]:
            context["error"] = "This run has no recorded attempts."
        else:
            attempts = data["attempts"]
            candidates = [_candidate_summary(g) for g in _group_candidates(attempts)]
            best_index = _select_best(candidates)
            candidates[0]["is_seed"] = True
            candidates[best_index]["is_best"] = True
            seed, best = candidates[0], candidates[best_index]
            context.update(
                seed=seed,
                best=best,
                candidates=candidates,
                num_attempts=len(attempts),
                num_candidates=len(candidates),
                best_is_seed=(best_index == 0),
                deltas={
                    key: _delta(best, seed, key)
                    for key in (
                        "mean_score",
                        "mean_entropy",
                        "fragile_count",
                        "boundary_count",
                        "confident_count",
                        "mean_input_tokens",
                        "mean_output_tokens",
                        "mean_cost_usd",
                    )
                },
            )

    return request.app.state.templates.TemplateResponse(request, "before_after.html", context)


@router.get("/cost", response_class=HTMLResponse)
async def cost_view(request: Request):
    summary = await _collector(request).get_cost_summary()
    return request.app.state.templates.TemplateResponse(
        request,
        "cost_view.html",
        {
            "nodes": summary["nodes"],
            "shared_prompt_blocks": summary["shared_prompt_blocks"],
            "scanned_traces": summary["scanned_traces"],
        },
    )
