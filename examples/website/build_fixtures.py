"""
Build the static website demo's frozen bundle from the use cases.

Default (offline): derive entropy/stability/attribution/counterfactual from the
hand-authored route labels using the SDK's own ``routing_entropy`` and the
analyser's fixed attribution table. Deterministic, no LLM, no cost. This is
what the committed ``web/data/demo.json`` is built from.

``--live`` is not supported for use cases (they are hand-authored incidents);
use the generic fixture harness for live capture.

Usage:
    python examples/website/build_fixtures.py --out web/data/demo.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from use_cases import USE_CASES  # noqa: E402

from conntrail.analyser import DivergenceAnalyser  # noqa: E402
from conntrail.contrast import ContrastSet  # noqa: E402
from conntrail.record import TraceRecord  # noqa: E402
from conntrail.utils.entropy import routing_entropy  # noqa: E402

_ANALYSER = DivergenceAnalyser()
_OVERHEAD = {
    "input_tokens": 1230,
    "output_tokens": 210,
    "total_tokens": 1440,
    "retries": 0,
    "latency_ms": 740.0,
    "cost_usd": 0.00041,
}


def _entropy(original_route: str, variant_routes: dict[str, str]) -> float:
    routes = [
        original_route,
        variant_routes["similar"],
        variant_routes["neutral"],
        variant_routes["opposite"],
    ]
    # normalise -0.0 so the page never renders "-0.00"
    return round(routing_entropy(routes), 3) + 0.0


def _routing_view(detection: dict[str, Any], fix: dict[str, Any]) -> dict[str, Any]:
    """Derive the analysis view for a routing-detection case (before + after)."""
    original_route = detection["original_route"]
    variant_routes = detection["variant_routes"]
    contrast_routes = {
        "similar": variant_routes["similar"],
        "neutral": variant_routes["neutral"],
        "opposite": variant_routes["opposite"],
    }

    before_entropy = _entropy(original_route, variant_routes)
    attribution, counterfactual = _ANALYSER._infer_attribution(original_route, contrast_routes)

    after_route = fix.get("after_route", original_route)
    after_variant_routes = fix["variant_routes"]
    after_entropy = _entropy(after_route, after_variant_routes)
    _, after_counterfactual = _ANALYSER._infer_attribution(
        after_route,
        {
            "similar": after_variant_routes["similar"],
            "neutral": after_variant_routes["neutral"],
            "opposite": after_variant_routes["opposite"],
        },
    )

    return {
        "before": {
            "route": original_route,
            "entropy_score": before_entropy,
            "stability": TraceRecord.stability_label(before_entropy),
            "attribution_dimension": attribution,
            "counterfactual_route": counterfactual,
            "variant_routes": contrast_routes,
        },
        "after": {
            "route": after_route,
            "entropy_score": after_entropy,
            "stability": TraceRecord.stability_label(after_entropy),
            "counterfactual_route": after_counterfactual,
            "variant_routes": {
                "similar": after_variant_routes["similar"],
                "neutral": after_variant_routes["neutral"],
                "opposite": after_variant_routes["opposite"],
            },
        },
    }


def _routing_record(use_case: dict[str, Any], view: dict[str, Any]) -> TraceRecord:
    """A TraceRecord for the incident's routing decision (for the CLI/gate tooling)."""
    detection = use_case["detection"]
    before = view["before"]
    labels = before["variant_routes"]
    stability = before["stability"]
    return TraceRecord(
        trace_id=TraceRecord.make_id(),
        node_id=detection["node_id"],
        timestamp=datetime.now(UTC),
        original_input=detection["original_input"],
        original_route=before["route"],
        entropy_score=before["entropy_score"],
        stability=stability,
        attribution_dimension=before["attribution_dimension"],
        plain_language_summary=TraceRecord.build_summary(
            detection["node_id"],
            before["route"],
            stability,
            before["entropy_score"],
            before["attribution_dimension"],
            before["counterfactual_route"],
        ),
        raw_contrasts=ContrastSet(
            similar=detection["contrasts"]["similar"],
            neutral=detection["contrasts"]["neutral"],
            opposite=detection["contrasts"]["opposite"],
        ),
        raw_outputs=dict(labels),
        counterfactual_route=before["counterfactual_route"],
        analysis_overhead=dict(_OVERHEAD),
    )


def build_use_cases() -> list[dict[str, Any]]:
    """Assemble the rich per-use-case view model the static page consumes."""
    out: list[dict[str, Any]] = []
    for use_case in USE_CASES:
        case = dict(use_case)
        detection = case["detection"]
        if detection["kind"] == "routing":
            view = _routing_view(detection, case["fix"])
            case["analysis"] = view
            case["delta"] = {
                "entropy_before": view["before"]["entropy_score"],
                "entropy_after": view["after"]["entropy_score"],
                "stability_before": view["before"]["stability"],
                "stability_after": view["after"]["stability"],
            }
        else:
            case["analysis"] = {
                "kind": "failure",
                "failure_category": detection["failure_category"],
                "error_type": detection.get("error_type"),
                "error_message": detection.get("error_message"),
                "affected_nodes": detection.get("affected_nodes", []),
            }
        out.append(case)
    return out


def build_records(use_cases: list[dict[str, Any]]) -> list[TraceRecord]:
    """Flat TraceRecords for routing cases (CLI/gate tooling + schema tests)."""
    records = []
    for case in use_cases:
        if case["detection"]["kind"] == "routing":
            records.append(_routing_record(case, case["analysis"]))
    return records


def build() -> list[TraceRecord]:
    """Harness entry point for `conntrail fixtures --harness ...:build`."""
    return build_records(build_use_cases())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the website demo fixture bundle")
    parser.add_argument("--out", type=Path, default=Path("web/data/demo.json"))
    args = parser.parse_args()

    use_cases = build_use_cases()
    records = [r.to_dict() for r in build_records(use_cases)]
    bundle = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "offline",
        "use_cases": use_cases,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(use_cases)} use case(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
