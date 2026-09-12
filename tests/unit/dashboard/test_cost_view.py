"""
Tests for the dashboard cost view (/cost) — per-node cost aggregation and
cross-node shared instruction blocks, against a mocked collector client.
"""
from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from conntrail_dashboard.app import create_app


class FakeCollectorClient:
    def __init__(self, cost_summary: dict[str, Any] | None = None) -> None:
        self.cost_summary_data = cost_summary or {"nodes": [], "shared_prompt_blocks": [], "scanned_traces": 0}

    async def list_traces(self, **params):  # pragma: no cover - unused by this page
        return {"traces": [], "limit": 50, "offset": 0}

    async def get_trace(self, trace_id):  # pragma: no cover - unused by this page
        return None

    async def get_gepa_attempts(self, run_id):  # pragma: no cover - unused by this page
        return None

    async def get_cost_summary(self) -> dict[str, Any]:
        return self.cost_summary_data


def _client_with(cost_summary):
    app = create_app(collector_client=FakeCollectorClient(cost_summary))
    return TestClient(app)


def test_cost_view_renders_node_aggregates():
    summary = {
        "nodes": [
            {
                "node_id": "router",
                "trace_count": 12,
                "llm_call_count": 24,
                "input_tokens": 48000,
                "output_tokens": 1200,
                "cached_input_tokens": 36000,
                "cache_write_tokens": 0,
                "cache_hit_ratio": 0.75,
                "total_cost_usd": 0.015,
                "mean_cost_usd": 0.00125,
                "mean_latency_ms": 240.5,
                "cost_warning_count": 3,
            }
        ],
        "shared_prompt_blocks": [
            {"hash": "abc123", "node_ids": ["router", "planner"], "occurrences": 9}
        ],
        "scanned_traces": 12,
    }
    client = _client_with(summary)
    resp = client.get("/cost")
    assert resp.status_code == 200
    body = resp.text
    assert "router" in body
    assert "75%" in body
    assert "$0.0150" in body
    assert "240.5 ms" in body
    assert "abc123" in body
    assert "router, planner" in body


def test_cost_view_empty_state():
    client = _client_with({"nodes": [], "shared_prompt_blocks": [], "scanned_traces": 0})
    resp = client.get("/cost")
    assert resp.status_code == 200
    assert "No cost telemetry yet" in resp.text


def test_cost_view_hides_shared_blocks_section_when_none():
    summary = {
        "nodes": [],
        "shared_prompt_blocks": [],
        "scanned_traces": 0,
    }
    client = _client_with(summary)
    body = client.get("/cost").text
    assert "Shared instruction blocks" not in body


def test_cost_view_renders_dash_for_missing_optionals():
    summary = {
        "nodes": [
            {
                "node_id": "plain",
                "trace_count": 1,
                "llm_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_tokens": 0,
                "cache_hit_ratio": None,
                "total_cost_usd": 0.0,
                "mean_cost_usd": None,
                "mean_latency_ms": None,
                "cost_warning_count": 0,
            }
        ],
        "shared_prompt_blocks": [],
        "scanned_traces": 1,
    }
    client = _client_with(summary)
    body = client.get("/cost").text
    assert "—" in body
