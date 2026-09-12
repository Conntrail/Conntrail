"""
Tests for GET /v1/cost-summary — the collector's per-node cost aggregation.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conntrail_server.app import create_app
from tests.fixtures.trace_records import make_trace_payload


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("COLLECTOR_API_KEY", raising=False)
    app = create_app(tmp_path / "traces.sqlite3")
    with TestClient(app) as c:
        yield c


def _usage(**overrides) -> dict:
    base = {
        "input_tokens": 2000,
        "output_tokens": 50,
        "cached_input_tokens": 1500,
        "cache_write_tokens": 0,
        "llm_call_count": 2,
        "models": ["claude-haiku-4-5-20251001"],
        "prompt_hashes": ["h1"],
    }
    base.update(overrides)
    return base


def test_cost_summary_requires_auth(tmp_path):
    import os

    os.environ["COLLECTOR_API_KEY"] = "secret"
    try:
        app = create_app(tmp_path / "auth.sqlite3")
        with TestClient(app) as c:
            resp = c.get("/v1/cost-summary")
            assert resp.status_code == 401
            resp = c.get("/v1/cost-summary", headers={"X-API-Key": "secret"})
            assert resp.status_code == 200
    finally:
        del os.environ["COLLECTOR_API_KEY"]


def test_empty_cost_summary(client):
    resp = client.get("/v1/cost-summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["nodes"] == []
    assert body["shared_prompt_blocks"] == []
    assert body["scanned_traces"] == 0


def test_cost_summary_aggregates_ingested_traces(client):
    for cost in (0.004, 0.008):
        client.post(
            "/v1/traces",
            json=make_trace_payload(
                node_id="router",
                token_usage=_usage(),
                cost_usd=cost,
                latency_ms=120.0,
                cost_findings=[
                    {"dimension": "cache_efficiency", "severity": "warning", "evidence": "e", "recommendation": "r"}
                ],
            ),
        )

    resp = client.get("/v1/cost-summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["scanned_traces"] == 2
    assert len(body["nodes"]) == 1
    node = body["nodes"][0]
    assert node["node_id"] == "router"
    assert node["trace_count"] == 2
    assert node["input_tokens"] == 4000
    assert node["cached_input_tokens"] == 3000
    assert node["cache_hit_ratio"] == 0.75
    assert node["total_cost_usd"] == 0.012
    assert node["mean_cost_usd"] == 0.006
    assert node["mean_latency_ms"] == 120.0
    assert node["cost_warning_count"] == 2


def test_cost_summary_shared_blocks_across_nodes(client):
    client.post(
        "/v1/traces",
        json=make_trace_payload(node_id="a", token_usage=_usage(prompt_hashes=["shared", "only-a"])),
    )
    client.post(
        "/v1/traces",
        json=make_trace_payload(node_id="b", token_usage=_usage(prompt_hashes=["shared"])),
    )

    body = client.get("/v1/cost-summary").json()
    shared = {b["hash"]: b for b in body["shared_prompt_blocks"]}
    assert shared["shared"]["node_ids"] == ["a", "b"]
    assert shared["shared"]["occurrences"] == 2
    assert "only-a" not in shared


def test_traces_without_cost_telemetry_are_counted_but_blank(client):
    client.post("/v1/traces", json=make_trace_payload(node_id="plain"))
    body = client.get("/v1/cost-summary").json()
    node = body["nodes"][0]
    assert node["trace_count"] == 1
    assert node["input_tokens"] == 0
    assert node["mean_cost_usd"] is None
    assert node["cache_hit_ratio"] is None
