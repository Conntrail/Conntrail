"""
Tests for the before/after CPE-GEPA panel (D2), against a mocked collector
HTTP client (a fixture pair of GEPA attempt records — the G3 query response
shape).
"""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from conntrail_dashboard.app import create_app


class FakeCollectorClient:
    """Test double standing in for CollectorClient — no real HTTP calls."""

    def __init__(self, gepa_runs: dict[str, dict[str, Any]] | None = None) -> None:
        self.gepa_runs = gepa_runs or {}

    async def list_traces(self, **params):  # pragma: no cover - unused by this page
        return {"traces": [], "limit": 50, "offset": 0}

    async def get_trace(self, trace_id):  # pragma: no cover - unused by this page
        return None

    async def get_gepa_attempts(self, run_id: str) -> dict[str, Any] | None:
        return self.gepa_runs.get(run_id)


def _trace(entropy: float, stability: str, attribution: str) -> dict[str, Any]:
    return {
        "trace_id": "t-1",
        "node_id": "classify_query",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "original_input": "hello",
        "original_route": "refund",
        "entropy_score": entropy,
        "stability": stability,
        "attribution_dimension": attribution,
        "plain_language_summary": "summary",
        "raw_contrasts": {"similar": "a", "neutral": "b", "opposite": "c"},
        "raw_outputs": {},
        "counterfactual_route": None,
        "status": "ok",
        "error_type": None,
        "error_message": None,
        "failure_category": "none",
    }


def _attempt(
    attempt_id: str,
    prompt: str,
    scalar_score: float | None,
    traces: list,
    token_usage: dict | None = None,
    cost_usd: float | None = None,
    latency_ms: float | None = None,
) -> dict:
    return {
        "run_id": "run-1",
        "attempt_id": attempt_id,
        "prompt_candidate": prompt,
        "scalar_score": scalar_score,
        "traces": traces,
        "token_usage": token_usage,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
    }


@pytest.fixture
def client_with():
    def _make(gepa_runs=None):
        collector = FakeCollectorClient(gepa_runs)
        app = create_app(collector_client=collector)
        return TestClient(app), collector

    return _make


def test_no_run_id_shows_lookup_prompt(client_with):
    client, _ = client_with()
    resp = client.get("/before-after")
    assert resp.status_code == 200
    assert "Enter a run_id" in resp.text


def test_unknown_run_id_shows_error(client_with):
    client, _ = client_with()
    resp = client.get("/before-after", params={"run_id": "does-not-exist"})
    assert resp.status_code == 200
    assert "No GEPA run found" in resp.text


def test_run_with_no_attempts_shows_error(client_with):
    client, _ = client_with({"run-1": {"run_id": "run-1", "attempts": []}})
    resp = client.get("/before-after", params={"run_id": "run-1"})
    assert "no recorded attempts" in resp.text


def test_renders_first_and_last_attempt_stats(client_with):
    first_traces = [
        _trace(0.9, "fragile", "semantic intensity"),
        _trace(0.8, "fragile", "semantic intensity"),
        _trace(0.1, "confident", "surface form"),
    ]
    last_traces = [
        _trace(0.1, "confident", "urgency/sentiment"),
        _trace(0.2, "confident", "urgency/sentiment"),
    ]
    run = {
        "run_id": "run-1",
        "attempts": [
            _attempt("a0", "Original prompt.", 0.0, first_traces),
            _attempt("a1", "Original prompt.", 0.0, [first_traces[0]]),
            _attempt("a2", "Optimized prompt.", 1.0, last_traces),
        ],
    }
    client, _ = client_with({"run-1": run})
    resp = client.get("/before-after", params={"run_id": "run-1"})
    assert resp.status_code == 200
    body = resp.text

    assert "Original prompt." in body
    assert "Optimized prompt." in body
    assert "3" in body  # num_attempts


def test_computed_stats_and_deltas_match_expected_values(client_with):
    """Assert against the actual computed numbers, not just presence."""
    first_traces = [
        _trace(0.9, "fragile", "semantic intensity"),
        _trace(0.7, "boundary", "semantic intensity"),
        _trace(0.1, "confident", "surface form"),
    ]
    # mean_entropy = (0.9+0.7+0.1)/3 = 0.5667, fragile=1, boundary=1, confident=1
    last_traces = [
        _trace(0.1, "confident", "urgency/sentiment"),
        _trace(0.2, "confident", "urgency/sentiment"),
    ]
    # mean_entropy = 0.15, fragile=0, boundary=0, confident=2

    run = {
        "run_id": "run-1",
        "attempts": [
            _attempt("a0", "Original prompt.", 0.25, first_traces),
            _attempt("a1", "Optimized prompt.", 0.9, last_traces),
        ],
    }
    client, _ = client_with({"run-1": run})

    from conntrail_dashboard.routes import _attempt_summary, _delta

    first_summary = _attempt_summary(run["attempts"][0])
    last_summary = _attempt_summary(run["attempts"][1])

    assert first_summary["mean_entropy"] == pytest.approx((0.9 + 0.7 + 0.1) / 3)
    assert first_summary["fragile_count"] == 1
    assert first_summary["boundary_count"] == 1
    assert first_summary["confident_count"] == 1
    assert first_summary["dominant_attribution"] == "semantic intensity"

    assert last_summary["mean_entropy"] == pytest.approx(0.15)
    assert last_summary["fragile_count"] == 0
    assert last_summary["confident_count"] == 2

    assert _delta(last_summary, first_summary, "mean_entropy") == pytest.approx(
        0.15 - (0.9 + 0.7 + 0.1) / 3
    )
    assert _delta(last_summary, first_summary, "fragile_count") == -1

    resp = client.get("/before-after", params={"run_id": "run-1"})
    body = resp.text
    assert "0.567" in body  # first mean entropy, formatted %.3f
    assert "0.150" in body  # last mean entropy
    assert "+0.650" in body  # scalar_score delta: 0.9 - 0.25


def test_missing_run_id_query_param_treated_as_no_run_id(client_with):
    client, _ = client_with()
    resp = client.get("/before-after")
    assert resp.status_code == 200


def test_attempt_cost_totals_and_deltas(client_with):
    first = _attempt(
        "a0", "Original prompt.", 0.25, [_trace(0.9, "fragile", "semantic intensity")],
        token_usage={"input_tokens": 1000, "output_tokens": 100}, cost_usd=0.010,
    )
    last = _attempt(
        "a1", "Optimized prompt.", 0.9, [_trace(0.1, "confident", "urgency/sentiment")],
        token_usage={"input_tokens": 600, "output_tokens": 60}, cost_usd=0.006,
    )
    run = {"run_id": "run-1", "attempts": [first, last]}
    client, _ = client_with({"run-1": run})

    from conntrail_dashboard.routes import _attempt_summary, _delta

    first_summary = _attempt_summary(first)
    assert first_summary["total_input_tokens"] == 1000
    assert first_summary["total_output_tokens"] == 100
    assert first_summary["total_cost_usd"] == 0.010

    last_summary = _attempt_summary(last)
    assert last_summary["total_input_tokens"] == 600
    assert _delta(last_summary, first_summary, "total_cost_usd") == pytest.approx(-0.004)

    resp = client.get("/before-after", params={"run_id": "run-1"})
    body = resp.text
    assert "1000 / 100" in body
    assert "600 / 60" in body
    assert "-0.004000" in body  # cost delta, %+.6f


def test_attempt_cost_falls_back_to_embedded_traces(client_with):
    rich_trace = dict(
        _trace(0.2, "confident", "urgency/sentiment"),
        token_usage={"input_tokens": 300, "output_tokens": 30},
        cost_usd=0.003,
        latency_ms=80.0,
    )
    attempt = _attempt("a0", "Prompt.", 0.5, [rich_trace])
    run = {"run_id": "run-1", "attempts": [attempt]}
    client, _ = client_with({"run-1": run})

    from conntrail_dashboard.routes import _attempt_summary

    summary = _attempt_summary(attempt)
    assert summary["total_input_tokens"] == 300
    assert summary["total_output_tokens"] == 30
    assert summary["total_cost_usd"] == pytest.approx(0.003)
    assert summary["mean_latency_ms"] == 80.0
