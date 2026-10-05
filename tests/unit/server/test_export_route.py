"""
Tests for GET /v1/export — the report-generation bundle.
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


def test_export_requires_auth(tmp_path):
    import os

    os.environ["COLLECTOR_API_KEY"] = "secret"
    try:
        app = create_app(tmp_path / "auth.sqlite3")
        with TestClient(app) as c:
            assert c.get("/v1/export").status_code == 401
            assert c.get("/v1/export", headers={"X-API-Key": "secret"}).status_code == 200
    finally:
        del os.environ["COLLECTOR_API_KEY"]


def test_export_empty_bundle(client):
    body = client.get("/v1/export").json()
    assert body["cost_summary"]["nodes"] == []
    assert body["traces"] == []
    assert body["generated_at"]
    assert body["filters"]["limit"] == 1000


def test_export_includes_summary_and_traces(client):
    for cost in (0.004, 0.008):
        client.post(
            "/v1/traces",
            json=make_trace_payload(
                node_id="router",
                token_usage=_usage(),
                cost_usd=cost,
                latency_ms=120.0,
                cost_findings=[
                    {
                        "dimension": "cache_efficiency",
                        "severity": "warning",
                        "evidence": "e",
                        "recommendation": "r",
                    }
                ],
            ),
        )

    body = client.get("/v1/export").json()
    assert body["cost_summary"]["scanned_traces"] == 2
    assert len(body["traces"]) == 2
    assert all(t["node_id"] == "router" for t in body["traces"])
    assert body["traces"][0]["cost_findings"][0]["dimension"] == "cache_efficiency"


def test_export_filters_by_node(client):
    client.post("/v1/traces", json=make_trace_payload(node_id="a", token_usage=_usage()))
    client.post("/v1/traces", json=make_trace_payload(node_id="b"))
    body = client.get("/v1/export", params={"node_id": "a"}).json()
    assert len(body["traces"]) == 1
    assert body["traces"][0]["node_id"] == "a"
