"""
Tests for conntrail_reports: view-model derivation and HTML rendering.
"""
from __future__ import annotations

import pytest

from conntrail_reports import (
    audit_data_from_bundle,
    build_audit_data,
    render_html,
)


def _node(node_id="router", **overrides):
    base = {
        "node_id": node_id,
        "trace_count": 2,
        "llm_call_count": 4,
        "input_tokens": 4000,
        "output_tokens": 100,
        "cached_input_tokens": 3000,
        "cache_write_tokens": 0,
        "total_cost_usd": 0.012,
        "mean_cost_usd": 0.006,
        "mean_latency_ms": 120.0,
        "cache_hit_ratio": 0.75,
        "cost_warning_count": 1,
    }
    base.update(overrides)
    return base


def _trace(node_id="router", findings=None):
    return {
        "node_id": node_id,
        "cost_findings": findings
        or [
            {
                "dimension": "cache_efficiency",
                "severity": "warning",
                "evidence": "low hit rate",
                "recommendation": "enable caching",
            },
            {
                "dimension": "observer_overhead",
                "severity": "info",
                "evidence": "observer cost measured",
                "recommendation": "tune sample_rate",
            },
        ],
    }


def test_totals_aggregate_nodes():
    data = build_audit_data(
        cost_summary={"nodes": [_node("a"), _node("b", input_tokens=1000, cached_input_tokens=0, output_tokens=50, total_cost_usd=0.001, cost_warning_count=0)], "shared_prompt_blocks": []},
        traces=[],
    )
    assert data.scanned_traces == 4
    assert data.totals["input_tokens"] == 5000
    assert data.totals["output_tokens"] == 150
    assert data.totals["total_cost_usd"] == pytest.approx(0.013)
    assert data.totals["cache_hit_ratio"] == pytest.approx(3000 / 5000, abs=1e-3)
    assert data.totals["warning_count"] == 1


def test_finding_rollup_counts_and_examples():
    data = build_audit_data(
        cost_summary={"nodes": [], "shared_prompt_blocks": []},
        traces=[_trace("router"), _trace("classifier")],
    )
    by_dim = {f.dimension: f for f in data.findings}
    assert by_dim["cache_efficiency"].warning_count == 2
    assert by_dim["observer_overhead"].info_count == 2
    assert len(by_dim["cache_efficiency"].examples) == 2
    assert by_dim["cache_efficiency"].examples[0]["node_id"] == "router"
    # dimensions with no findings still appear (derived-not-invented table)
    assert by_dim["prompt_size"].total == 0


def test_rollup_ignores_unknown_dimensions():
    data = build_audit_data(
        cost_summary={"nodes": [], "shared_prompt_blocks": []},
        traces=[_trace(findings=[{"dimension": "made_up", "severity": "warning", "evidence": "x", "recommendation": "y"}])],
    )
    assert all(f.dimension != "made_up" for f in data.findings)
    assert sum(f.total for f in data.findings) == 0


def test_audit_data_from_bundle_requires_cost_summary():
    with pytest.raises(ValueError):
        audit_data_from_bundle({"records": []})


def test_audit_data_from_bundle_round_trip():
    data = audit_data_from_bundle(
        {"cost_summary": {"nodes": [_node()], "shared_prompt_blocks": [], "scanned_traces": 2}, "traces": [_trace()]}
    )
    assert data.source == "bundle"
    assert data.scanned_traces == 2


def test_render_html_is_self_contained_and_reports_disclaimer():
    data = build_audit_data(
        cost_summary={
            "nodes": [_node()],
            "shared_prompt_blocks": [{"hash": "abc123", "node_ids": ["a", "b"], "occurrences": 2}],
        },
        traces=[_trace()],
    )
    html = render_html(data)
    assert "<html" in html
    assert "LLM cost audit" in html
    assert "not an invoice" in html
    assert "abc123" in html
    assert "cache_efficiency" in html
    # embedded stylesheet, no external references
    assert "<style>" in html
    assert "http://" not in html and "https://" not in html
