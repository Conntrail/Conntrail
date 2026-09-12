"""
Tests for POST/GET /v1/gepa-attempts (G3).
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


def _attempt_payload(*, run_id="run-1", attempt_id="attempt-1", scalar_score=0.5, num_traces=1):
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "prompt_candidate": "Classify the message into a category.",
        "scalar_score": scalar_score,
        "traces": [make_trace_payload() for _ in range(num_traces)],
    }


def test_post_returns_201_with_attempt_id(client):
    payload = _attempt_payload()
    resp = client.post("/v1/gepa-attempts", json=payload)
    assert resp.status_code == 201
    assert resp.json() == {"attempt_id": "attempt-1"}


def test_get_by_run_id_round_trips(client):
    payload = _attempt_payload(num_traces=2)
    client.post("/v1/gepa-attempts", json=payload)

    resp = client.get("/v1/gepa-attempts", params={"run_id": "run-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run-1"
    assert len(body["attempts"]) == 1
    attempt = body["attempts"][0]
    assert attempt["attempt_id"] == "attempt-1"
    assert attempt["prompt_candidate"] == payload["prompt_candidate"]
    assert attempt["scalar_score"] == 0.5
    assert len(attempt["traces"]) == 2
    assert attempt["traces"][0]["raw_contrasts"] == payload["traces"][0]["raw_contrasts"]


def test_get_filters_by_run_id(client):
    client.post("/v1/gepa-attempts", json=_attempt_payload(run_id="run-a", attempt_id="a1"))
    client.post("/v1/gepa-attempts", json=_attempt_payload(run_id="run-b", attempt_id="b1"))

    resp = client.get("/v1/gepa-attempts", params={"run_id": "run-a"})
    body = resp.json()
    assert [a["attempt_id"] for a in body["attempts"]] == ["a1"]


def test_get_preserves_insertion_order_within_a_run(client):
    for i in range(3):
        client.post(
            "/v1/gepa-attempts", json=_attempt_payload(run_id="run-1", attempt_id=f"a{i}")
        )
    resp = client.get("/v1/gepa-attempts", params={"run_id": "run-1"})
    body = resp.json()
    assert [a["attempt_id"] for a in body["attempts"]] == ["a0", "a1", "a2"]


def test_get_unknown_run_id_returns_404(client):
    resp = client.get("/v1/gepa-attempts", params={"run_id": "does-not-exist"})
    assert resp.status_code == 404


def test_reposting_same_attempt_id_updates_in_place(client):
    """GEPA can legitimately re-score the same attempt (e.g. re-evaluating an
    accepted candidate against the full valset) — a second POST with the
    same attempt_id must not error, and must reflect the latest score."""
    client.post("/v1/gepa-attempts", json=_attempt_payload(scalar_score=0.0))
    resp = client.post("/v1/gepa-attempts", json=_attempt_payload(scalar_score=1.0))
    assert resp.status_code == 201

    body = client.get("/v1/gepa-attempts", params={"run_id": "run-1"}).json()
    assert len(body["attempts"]) == 1
    assert body["attempts"][0]["scalar_score"] == 1.0


def test_null_scalar_score_round_trips(client):
    payload = _attempt_payload(scalar_score=None)
    client.post("/v1/gepa-attempts", json=payload)
    resp = client.get("/v1/gepa-attempts", params={"run_id": "run-1"})
    assert resp.json()["attempts"][0]["scalar_score"] is None


def test_attempt_cost_fields_round_trip(client):
    payload = _attempt_payload()
    payload["token_usage"] = {"input_tokens": 400, "output_tokens": 40}
    payload["cost_usd"] = 0.002
    payload["latency_ms"] = 33.0
    resp = client.post("/v1/gepa-attempts", json=payload)
    assert resp.status_code == 201

    fetched = client.get("/v1/gepa-attempts", params={"run_id": "run-1"}).json()
    attempt = fetched["attempts"][0]
    assert attempt["token_usage"] == {"input_tokens": 400, "output_tokens": 40}
    assert attempt["cost_usd"] == 0.002
    assert attempt["latency_ms"] == 33.0


def test_attempt_without_cost_fields_defaults_to_none(client):
    client.post("/v1/gepa-attempts", json=_attempt_payload())
    fetched = client.get("/v1/gepa-attempts", params={"run_id": "run-1"}).json()
    attempt = fetched["attempts"][0]
    assert attempt["token_usage"] is None
    assert attempt["cost_usd"] is None
    assert attempt["latency_ms"] is None
