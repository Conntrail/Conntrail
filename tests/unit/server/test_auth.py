"""
Tests for the collector's X-API-Key auth gate (T4).
"""
from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from conntrail_server.app import create_app
from tests.fixtures.trace_records import make_trace_payload


def _make_client(tmp_path) -> TestClient:
    app = create_app(tmp_path / "traces.sqlite3")
    return TestClient(app)


# ---------------------------------------------------------------------------
# COLLECTOR_API_KEY set — enforced
# ---------------------------------------------------------------------------


def test_request_without_key_returns_401(tmp_path, monkeypatch):
    monkeypatch.setenv("COLLECTOR_API_KEY", "secret-123")
    with _make_client(tmp_path) as client:
        resp = client.get("/v1/traces")
        assert resp.status_code == 401


def test_request_with_wrong_key_returns_401(tmp_path, monkeypatch):
    monkeypatch.setenv("COLLECTOR_API_KEY", "secret-123")
    with _make_client(tmp_path) as client:
        resp = client.get("/v1/traces", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401


def test_request_with_correct_key_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("COLLECTOR_API_KEY", "secret-123")
    with _make_client(tmp_path) as client:
        resp = client.get("/v1/traces", headers={"X-API-Key": "secret-123"})
        assert resp.status_code == 200


def test_ingest_requires_key_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv("COLLECTOR_API_KEY", "secret-123")
    with _make_client(tmp_path) as client:
        payload = make_trace_payload()

        rejected = client.post("/v1/traces", json=payload)
        assert rejected.status_code == 401

        accepted = client.post(
            "/v1/traces", json=payload, headers={"X-API-Key": "secret-123"}
        )
        assert accepted.status_code == 201


def test_trace_detail_route_requires_key_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv("COLLECTOR_API_KEY", "secret-123")
    with _make_client(tmp_path) as client:
        payload = make_trace_payload()
        client.post("/v1/traces", json=payload, headers={"X-API-Key": "secret-123"})

        resp = client.get(f"/v1/traces/{payload['trace_id']}")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# COLLECTOR_API_KEY unset — dev-mode passthrough + startup warning
# ---------------------------------------------------------------------------


def test_unset_key_allows_unauthenticated_requests(tmp_path, monkeypatch):
    monkeypatch.delenv("COLLECTOR_API_KEY", raising=False)
    with _make_client(tmp_path) as client:
        resp = client.get("/v1/traces")
        assert resp.status_code == 200


def test_unset_key_emits_startup_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("COLLECTOR_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger="conntrail_server"):
        create_app(tmp_path / "traces.sqlite3")
    assert any("COLLECTOR_API_KEY" in record.message for record in caplog.records)


def test_set_key_emits_no_startup_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("COLLECTOR_API_KEY", "secret-123")
    with caplog.at_level(logging.WARNING, logger="conntrail_server"):
        create_app(tmp_path / "traces.sqlite3")
    assert not any("COLLECTOR_API_KEY" in record.message for record in caplog.records)
