"""
Tests for the dashboard routes (D1), against a mocked collector HTTP client.
"""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from conntrail_dashboard.app import create_app


class FakeCollectorClient:
    """Test double standing in for CollectorClient — no real HTTP calls."""

    def __init__(self, traces: list[dict[str, Any]] | None = None) -> None:
        self.traces = traces or []
        self.list_calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []

    async def list_traces(self, **params: Any) -> dict[str, Any]:
        self.list_calls.append(params)
        rows = self.traces
        if params.get("node_id") is not None:
            rows = [t for t in rows if t["node_id"] == params["node_id"]]
        if params.get("stability") is not None:
            rows = [t for t in rows if t["stability"] == params["stability"]]
        if params.get("failure_category") is not None:
            rows = [t for t in rows if t.get("failure_category") == params["failure_category"]]
        limit = params.get("limit", 50)
        offset = params.get("offset", 0)
        page = rows[offset : offset + limit]
        return {"traces": page, "limit": limit, "offset": offset}

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        self.get_calls.append(trace_id)
        for t in self.traces:
            if t["trace_id"] == trace_id:
                return t
        return None


def _trace(**overrides: Any) -> dict[str, Any]:
    base = {
        "trace_id": "t-1",
        "node_id": "router",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "original_input": "I need a refund",
        "original_route": "refund",
        "entropy_score": 0.12,
        "stability": "confident",
        "attribution_dimension": "urgency",
        "plain_language_summary": "The 'router' node routed to 'refund'.",
        "raw_contrasts": {"similar": "a", "neutral": "b", "opposite": "c"},
        "raw_outputs": {"similar": "refund"},
        "counterfactual_route": "escalation",
        "status": "ok",
        "error_type": None,
        "error_message": None,
        "failure_category": "none",
    }
    base.update(overrides)
    return base


@pytest.fixture
def client_with():
    def _make(traces=None):
        collector = FakeCollectorClient(traces or [])
        app = create_app(collector_client=collector)
        return TestClient(app), collector

    return _make


# ---------------------------------------------------------------------------
# Trace list
# ---------------------------------------------------------------------------


def test_trace_list_renders_rows(client_with):
    client, _ = client_with([_trace(trace_id="t-1", node_id="router")])
    resp = client.get("/")
    assert resp.status_code == 200
    assert "router" in resp.text
    assert "t-1" in resp.text


def test_trace_list_empty_state(client_with):
    client, _ = client_with([])
    resp = client.get("/")
    assert resp.status_code == 200
    assert "No traces match" in resp.text


def test_trace_list_passes_filters_to_collector(client_with):
    client, collector = client_with([_trace()])
    resp = client.get("/", params={"node_id": "router", "stability": "fragile"})
    assert resp.status_code == 200
    assert collector.list_calls[-1]["node_id"] == "router"
    assert collector.list_calls[-1]["stability"] == "fragile"


def test_trace_list_htmx_request_returns_partial_only(client_with):
    client, _ = client_with([_trace()])
    resp = client.get("/", headers={"HX-Request": "true"})
    assert resp.status_code == 200
    assert "<html" not in resp.text.lower()
    assert "router" in resp.text


def test_trace_list_full_page_includes_layout(client_with):
    client, _ = client_with([_trace()])
    resp = client.get("/")
    assert "<html" in resp.text.lower()
    assert "Trace Explorer" in resp.text


# ---------------------------------------------------------------------------
# Trace detail
# ---------------------------------------------------------------------------


def test_trace_detail_renders_full_record(client_with):
    trace = _trace(
        trace_id="t-2",
        status="error",
        error_type="timeout",
        error_message="node exceeded 5s",
        failure_category="timeout",
    )
    client, _ = client_with([trace])
    resp = client.get("/traces/t-2")
    assert resp.status_code == 200
    assert "t-2" in resp.text
    assert "timeout" in resp.text
    assert "node exceeded 5s" in resp.text
    assert "urgency" in resp.text  # attribution_dimension
    assert "escalation" in resp.text  # counterfactual_route


def test_trace_detail_404_on_missing_trace(client_with):
    client, _ = client_with([])
    resp = client.get("/traces/does-not-exist")
    assert resp.status_code == 404
    assert "not found" in resp.text.lower()


# ---------------------------------------------------------------------------
# Failure view
# ---------------------------------------------------------------------------


def test_failure_view_shows_counts_per_category(client_with):
    traces = [
        _trace(trace_id="t-1", status="error", error_type="timeout", failure_category="timeout"),
        _trace(trace_id="t-2", status="error", error_type="timeout", failure_category="timeout"),
        _trace(
            trace_id="t-3",
            status="error",
            error_type="ValueError",
            failure_category="exception",
        ),
        _trace(trace_id="t-4"),  # failure_category="none" — not a failure
    ]
    client, _ = client_with(traces)
    resp = client.get("/failures")
    assert resp.status_code == 200
    assert ">2<" in resp.text  # timeout count
    assert ">1<" in resp.text  # exception count


def test_failure_view_filters_rows_by_selected_category(client_with):
    traces = [
        _trace(trace_id="t-1", status="error", error_type="timeout", failure_category="timeout"),
        _trace(
            trace_id="t-2",
            status="error",
            error_type="ValueError",
            failure_category="exception",
        ),
    ]
    client, _ = client_with(traces)
    resp = client.get("/failures", params={"failure_category": "timeout"})
    assert resp.status_code == 200
    assert "t-1" in resp.text
    assert "t-2" not in resp.text


def test_failure_view_no_category_selected_shows_no_rows(client_with):
    client, _ = client_with(
        [_trace(trace_id="t-1", status="error", error_type="timeout", failure_category="timeout")]
    )
    resp = client.get("/failures")
    assert resp.status_code == 200
    assert "Select a category" in resp.text


# ---------------------------------------------------------------------------
# /healthz — O3's compose healthcheck
# ---------------------------------------------------------------------------


def test_healthz(client_with):
    client, _ = client_with([])
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
