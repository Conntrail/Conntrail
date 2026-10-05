"""
Tests for the static website demo fixtures (web/data/demo.json).

Validates that the committed bundle conforms to the TraceRecord.to_dict()
contract and contains everything the static page needs to render — so the
demo can never silently ship a broken fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conntrail.record import TraceRecord

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE = _REPO_ROOT / "web" / "data" / "demo.json"


@pytest.fixture(scope="module")
def bundle() -> dict:
    assert _BUNDLE.exists(), f"missing demo fixture bundle: {_BUNDLE}"
    return json.loads(_BUNDLE.read_text(encoding="utf-8"))


def test_bundle_has_records(bundle):
    assert bundle["records"], "demo bundle has no records"
    assert bundle.get("generated_at")


def test_every_record_round_trips_through_trace_record(bundle):
    for record in bundle["records"]:
        rebuilt = TraceRecord.from_dict(record)
        assert rebuilt.to_dict()["original_route"] == record["original_route"]
        assert rebuilt.entropy_score == record["entropy_score"]


def test_records_are_demo_renderable(bundle):
    for record in bundle["records"]:
        assert record["scenario_id"]
        assert record["scenario_title"]
        assert record["original_input"]
        assert record["original_route"]
        assert 0.0 <= record["entropy_score"] <= 1.0
        assert record["stability"] in ("confident", "boundary", "fragile")
        # the page renders all three contrast variants and their routes
        for key in ("similar", "neutral", "opposite"):
            assert record["raw_contrasts"][key]
            assert record["raw_outputs"][key]
        assert record["candidate_routes"]
        assert record["original_route"] in record["candidate_routes"]


def test_stability_matches_entropy(bundle):
    for record in bundle["records"]:
        assert record["stability"] == TraceRecord.stability_label(record["entropy_score"])


def test_bundle_covers_all_stability_labels(bundle):
    labels = {r["stability"] for r in bundle["records"]}
    assert labels == {"confident", "boundary", "fragile"}
