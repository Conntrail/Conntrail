"""
Tests for GET /v1/traces and GET /v1/traces/{trace_id} (T3).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from conntrail_server.app import create_app
from tests.fixtures.trace_records import make_trace_payload


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("COLLECTOR_API_KEY", raising=False)  # these tests exercise dev-mode auth
    app = create_app(tmp_path / "traces.sqlite3")
    with TestClient(app) as c:
        yield c


def _seed(client, payload):
    resp = client.post("/v1/traces", json=payload)
    assert resp.status_code == 201
    return payload


# ---------------------------------------------------------------------------
# GET /v1/traces/{trace_id}
# ---------------------------------------------------------------------------


def test_get_unknown_trace_id_returns_404(client):
    resp = client.get("/v1/traces/does-not-exist")
    assert resp.status_code == 404


def test_get_known_trace_id_returns_full_record(client):
    payload = _seed(client, make_trace_payload())
    resp = client.get(f"/v1/traces/{payload['trace_id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["trace_id"] == payload["trace_id"]
    assert body["raw_contrasts"] == payload["raw_contrasts"]
    assert body["raw_outputs"] == payload["raw_outputs"]


# ---------------------------------------------------------------------------
# GET /v1/traces — filters
# ---------------------------------------------------------------------------


def test_list_with_no_filters_returns_all(client):
    p1 = _seed(client, make_trace_payload())
    p2 = _seed(client, make_trace_payload())
    resp = client.get("/v1/traces")
    assert resp.status_code == 200
    body = resp.json()
    ids = {t["trace_id"] for t in body["traces"]}
    assert ids == {p1["trace_id"], p2["trace_id"]}


def test_list_filters_by_node_id(client):
    p1 = _seed(client, make_trace_payload(node_id="router_a"))
    _seed(client, make_trace_payload(node_id="router_b"))
    resp = client.get("/v1/traces", params={"node_id": "router_a"})
    body = resp.json()
    assert [t["trace_id"] for t in body["traces"]] == [p1["trace_id"]]


def test_list_filters_by_stability(client):
    p1 = _seed(client, make_trace_payload(stability="fragile"))
    _seed(client, make_trace_payload(stability="confident"))
    resp = client.get("/v1/traces", params={"stability": "fragile"})
    body = resp.json()
    assert [t["trace_id"] for t in body["traces"]] == [p1["trace_id"]]


def test_list_filters_by_failure_category(client):
    p1 = _seed(client, make_trace_payload(status="error", error_type="timeout"))
    _seed(client, make_trace_payload(status="ok", original_route="refund"))
    resp = client.get("/v1/traces", params={"failure_category": "timeout"})
    body = resp.json()
    assert [t["trace_id"] for t in body["traces"]] == [p1["trace_id"]]


def test_list_filters_by_status(client):
    p1 = _seed(client, make_trace_payload(status="error"))
    _seed(client, make_trace_payload(status="ok"))
    resp = client.get("/v1/traces", params={"status": "error"})
    body = resp.json()
    assert [t["trace_id"] for t in body["traces"]] == [p1["trace_id"]]


def test_list_filters_by_since_and_until(client):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    p_old = _seed(client, make_trace_payload(timestamp=base - timedelta(days=2)))
    p_mid = _seed(client, make_trace_payload(timestamp=base))
    p_new = _seed(client, make_trace_payload(timestamp=base + timedelta(days=2)))

    resp = client.get(
        "/v1/traces",
        params={
            "since": (base - timedelta(days=1)).isoformat(),
            "until": (base + timedelta(days=1)).isoformat(),
        },
    )
    body = resp.json()
    ids = [t["trace_id"] for t in body["traces"]]
    assert ids == [p_mid["trace_id"]]
    assert p_old["trace_id"] not in ids
    assert p_new["trace_id"] not in ids


def test_list_combines_multiple_filters(client):
    match = _seed(client, make_trace_payload(node_id="router", stability="fragile"))
    _seed(client, make_trace_payload(node_id="other", stability="fragile"))
    _seed(client, make_trace_payload(node_id="router", stability="confident"))

    resp = client.get("/v1/traces", params={"node_id": "router", "stability": "fragile"})
    body = resp.json()
    assert [t["trace_id"] for t in body["traces"]] == [match["trace_id"]]


def test_list_empty_result(client):
    _seed(client, make_trace_payload(node_id="router"))
    resp = client.get("/v1/traces", params={"node_id": "does-not-exist"})
    assert resp.json()["traces"] == []


# ---------------------------------------------------------------------------
# GET /v1/traces — pagination
# ---------------------------------------------------------------------------


def test_list_pagination(client):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    payloads = [
        _seed(client, make_trace_payload(node_id="router", timestamp=base + timedelta(minutes=i)))
        for i in range(5)
    ]
    expected_order = [p["trace_id"] for p in reversed(payloads)]  # newest first

    resp = client.get("/v1/traces", params={"node_id": "router", "limit": 2, "offset": 2})
    body = resp.json()
    assert [t["trace_id"] for t in body["traces"]] == expected_order[2:4]
    assert body["limit"] == 2
    assert body["offset"] == 2


# ---------------------------------------------------------------------------
# response shape
# ---------------------------------------------------------------------------


def test_detail_response_deserialises_nested_json_not_double_encoded(client):
    payload = _seed(client, make_trace_payload())
    body = client.get(f"/v1/traces/{payload['trace_id']}").json()
    assert isinstance(body["raw_contrasts"], dict)
    assert isinstance(body["raw_outputs"], dict)
    assert body["raw_contrasts"]["similar"] == "a"
