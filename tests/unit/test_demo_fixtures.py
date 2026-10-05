"""
Tests for the static website demo fixtures (web/data/demo.json).

Validates that the committed bundle is internally consistent and renderable:
routing cases' entropy/stability must match the SDK's own formulas, failure
cases must carry valid categories, and the flat `records` list must round-trip
through TraceRecord.to_dict().
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conntrail.analyser import DivergenceAnalyser
from conntrail.record import TraceRecord
from conntrail.utils.entropy import routing_entropy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE = _REPO_ROOT / "web" / "data" / "demo.json"

_VALID_CATEGORIES = {"exception", "retry_loop", "timeout", "malformed_output", "none"}
_ANALYSER = DivergenceAnalyser()


@pytest.fixture(scope="module")
def bundle() -> dict:
    assert _BUNDLE.exists(), f"missing demo fixture bundle: {_BUNDLE}"
    return json.loads(_BUNDLE.read_text(encoding="utf-8"))


def test_bundle_has_use_cases_and_records(bundle):
    assert len(bundle["use_cases"]) >= 3
    assert bundle.get("generated_at")
    assert bundle["records"], "expected flat records for CLI/gate tooling"


def test_use_cases_have_required_fields(bundle):
    for case in bundle["use_cases"]:
        assert case["id"] and case["title"] and case["company"]
        assert case["tagline"] and case["incident"]
        assert case["surface"]["kind"] in {"web_chat", "whatsapp", "leads", "outage", "moderation"}
        assert case["detection"]["kind"] in {"routing", "failure"}
        assert case["fix"]["summary"]
        assert case["source_refs"], f"{case['id']} should cite the pattern's source"


def _surface_items(surface: dict) -> list:
    for key in ("transcript", "leads", "services", "queue"):
        if key in surface:
            return surface[key]
    return []


def test_use_cases_have_an_after_fix_surface(bundle):
    """The After-fix toggle shows the user-facing outcome, so it needs content."""
    for case in bundle["use_cases"]:
        after = case.get("surface_after")
        assert after, f"{case['id']} is missing surface_after"
        assert after["kind"] == case["surface"]["kind"]
        assert _surface_items(after), f"{case['id']} surface_after has no content"


def test_routing_cases_derive_entropy_and_stability(bundle):
    routing = [c for c in bundle["use_cases"] if c["detection"]["kind"] == "routing"]
    assert routing, "expected at least one routing case"
    for case in routing:
        detection = case["detection"]
        before = case["analysis"]["before"]
        after = case["analysis"]["after"]

        expected = round(
            routing_entropy(
                [
                    detection["original_route"],
                    detection["variant_routes"]["similar"],
                    detection["variant_routes"]["neutral"],
                    detection["variant_routes"]["opposite"],
                ]
            ),
            3,
        ) + 0.0
        assert before["entropy_score"] == pytest.approx(expected)
        assert before["stability"] == TraceRecord.stability_label(before["entropy_score"])
        assert after["stability"] == TraceRecord.stability_label(after["entropy_score"])
        # a fix should not make the decision less stable
        assert after["entropy_score"] <= before["entropy_score"] + 1e-9


def test_routing_attribution_matches_sdk_table(bundle):
    for case in bundle["use_cases"]:
        if case["detection"]["kind"] != "routing":
            continue
        before = case["analysis"]["before"]
        dim, cf = _ANALYSER._infer_attribution(before["route"], before["variant_routes"])
        assert before["attribution_dimension"] == dim
        assert before["counterfactual_route"] == cf


def test_failure_cases_use_valid_categories(bundle):
    failures = [c for c in bundle["use_cases"] if c["detection"]["kind"] == "failure"]
    assert failures, "expected at least one failure/outage case"
    for case in failures:
        analysis = case["analysis"]
        assert analysis["failure_category"] in _VALID_CATEGORIES
        assert analysis["affected_nodes"]
        for node in analysis["affected_nodes"]:
            assert node["failure_category"] in _VALID_CATEGORIES
            assert node["count"] > 0


def test_flat_records_round_trip(bundle):
    for record in bundle["records"]:
        rebuilt = TraceRecord.from_dict(record)
        assert rebuilt.to_dict()["original_route"] == record["original_route"]
        assert rebuilt.entropy_score == record["entropy_score"]


def test_use_cases_cover_both_detection_kinds(bundle):
    kinds = {c["detection"]["kind"] for c in bundle["use_cases"]}
    assert kinds == {"routing", "failure"}
