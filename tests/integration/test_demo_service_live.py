"""
Live integration tests for the demo service.

These hit real free-tier models (no mocks) and are skipped unless a provider
key is present. They are marked `integration` so the default unit run skips
them.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from demo.service.app import app
from demo.service.models import NoProviderKey, candidate_configs
from demo.service.runner import run_analysis

pytestmark = pytest.mark.integration

_HAVE_KEY = bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY"))
_skip = pytest.mark.skipif(not _HAVE_KEY, reason="needs GROQ_API_KEY or OPENROUTER_API_KEY")


@_skip
async def test_run_analysis_produces_a_trace_record():
    try:
        configs = candidate_configs()
    except NoProviderKey:
        pytest.skip("no free-tier key configured")

    res = await run_analysis("I want a refund for an order that never arrived.", configs)

    assert res["original_route"] in ("refund", "order_status", "escalate", "general")
    assert 0.0 <= res["entropy_score"] <= 1.0
    assert res["stability"] in ("confident", "boundary", "fragile")
    assert set(res["raw_contrasts"]) == {"similar", "neutral", "opposite"}
    assert set(res["raw_outputs"]) == {"similar", "neutral", "opposite"}
    assert res["analysis_overhead"]["cost_usd"] == 0.0  # free models only


@_skip
def test_analyse_endpoint_returns_and_caches():
    with TestClient(app) as client:
        payload = {"message": "This is the third time I've asked. Refund me or I call my lawyer."}
        first = client.post("/api/analyse", json=payload)
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["stability"] in ("confident", "boundary", "fragile")
        assert body["cached"] is False

        second = client.post("/api/analyse", json=payload)
        assert second.status_code == 200
        assert second.json()["cached"] is True


def test_config_endpoint_shape():
    # No key required for the config contract.
    with TestClient(app) as client:
        cfg = client.get("/api/config").json()
    assert "openrouter" in cfg["providers"]
    assert cfg["byok"] is True
    assert "daily_remaining" in cfg
