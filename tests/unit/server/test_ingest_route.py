"""
Tests for POST /v1/traces (T2).
"""
from __future__ import annotations

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


def test_valid_payload_returns_201_with_trace_id(client):
    payload = make_trace_payload()
    resp = client.post("/v1/traces", json=payload)
    assert resp.status_code == 201
    assert resp.json() == {"trace_id": payload["trace_id"]}


def test_valid_payload_row_appears_in_store_via_get(client):
    payload = make_trace_payload()
    client.post("/v1/traces", json=payload)

    resp = client.get(f"/v1/traces/{payload['trace_id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["trace_id"] == payload["trace_id"]
    assert body["original_route"] == payload["original_route"]


def test_missing_required_field_returns_422(client):
    payload = make_trace_payload()
    del payload["node_id"]
    resp = client.post("/v1/traces", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert any(err["loc"][-1] == "node_id" for err in detail)


def test_malformed_json_returns_422(client):
    resp = client.post(
        "/v1/traces",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_invalid_timestamp_returns_422(client):
    payload = make_trace_payload()
    payload["timestamp"] = "not-a-timestamp"
    resp = client.post("/v1/traces", json=payload)
    assert resp.status_code == 422


def test_extra_unknown_fields_do_not_error(client):
    payload = make_trace_payload()
    payload["some_future_field"] = "unrecognised but harmless"
    resp = client.post("/v1/traces", json=payload)
    assert resp.status_code == 201


def test_error_status_payload_round_trips(client):
    payload = make_trace_payload(status="error")
    resp = client.post("/v1/traces", json=payload)
    assert resp.status_code == 201

    fetched = client.get(f"/v1/traces/{payload['trace_id']}").json()
    assert fetched["status"] == "error"
    assert fetched["error_type"] == "ValueError"
    assert fetched["error_message"] == "boom"


# ---------------------------------------------------------------------------
# failure_category persistence (C2)
# ---------------------------------------------------------------------------


def test_ingest_persists_retry_loop_failure_category(client):
    payload = make_trace_payload(status="error", error_type="retry_loop")
    client.post("/v1/traces", json=payload)
    fetched = client.get(f"/v1/traces/{payload['trace_id']}").json()
    assert fetched["failure_category"] == "retry_loop"


def test_ingest_persists_timeout_failure_category(client):
    payload = make_trace_payload(status="error", error_type="timeout")
    client.post("/v1/traces", json=payload)
    fetched = client.get(f"/v1/traces/{payload['trace_id']}").json()
    assert fetched["failure_category"] == "timeout"


def test_ingest_persists_exception_failure_category(client):
    payload = make_trace_payload(status="error", error_type="ValueError")
    client.post("/v1/traces", json=payload)
    fetched = client.get(f"/v1/traces/{payload['trace_id']}").json()
    assert fetched["failure_category"] == "exception"


def test_ingest_persists_malformed_output_failure_category(client):
    payload = make_trace_payload(status="ok", original_route="unknown")
    client.post("/v1/traces", json=payload)
    fetched = client.get(f"/v1/traces/{payload['trace_id']}").json()
    assert fetched["failure_category"] == "malformed_output"


def test_ingest_persists_none_failure_category_for_clean_success(client):
    payload = make_trace_payload(status="ok", original_route="refund")
    client.post("/v1/traces", json=payload)
    fetched = client.get(f"/v1/traces/{payload['trace_id']}").json()
    assert fetched["failure_category"] == "none"


def test_ingest_ignores_client_supplied_failure_category(client):
    """failure_category is server-computed — a client-sent value is discarded."""
    payload = make_trace_payload(status="ok", original_route="refund")
    payload["failure_category"] = "timeout"  # client lying about its own category
    client.post("/v1/traces", json=payload)
    fetched = client.get(f"/v1/traces/{payload['trace_id']}").json()
    assert fetched["failure_category"] == "none"
