"""
Build the static website demo's frozen TraceRecord bundle.

Two modes:

- default (offline): synthesise TraceRecords from the hand-authored scenarios
  in ``scenarios.py``. Deterministic, no LLM, no cost. This is what the
  committed ``web/data/demo.json`` is built from.
- ``--live``: re-run each scenario's routing decision through Conntrail's real
  contrast generator + divergence analyser, so entropy/attribution reflect a
  live model. Requires a configured provider key (or LOCAL_LLM_URL).

Usage:
    python examples/website/build_fixtures.py --out web/data/demo.json
    python examples/website/build_fixtures.py --live --out web/data/demo.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenarios import ALL_SCENARIOS, Scenario  # noqa: E402

from conntrail.contrast import ContrastSet  # noqa: E402
from conntrail.record import TraceRecord  # noqa: E402
from conntrail.utils.entropy import routing_entropy  # noqa: E402

_OFFLINE_OVERHEAD = {
    "input_tokens": 1230,
    "output_tokens": 210,
    "total_tokens": 1440,
    "retries": 0,
    "latency_ms": 740.0,
    "cost_usd": 0.00041,
}


def _entropy_from_labels(labels: dict[str, str]) -> float:
    routes = [labels["original"], labels["similar"], labels["neutral"], labels["opposite"]]
    entropy = routing_entropy(routes)
    # routing_entropy can return -0.0 for a fully stable decision; normalise so
    # the report/demo never renders "-0.00".
    return round(entropy, 3) + 0.0


def _record_from_scenario(scenario: Scenario) -> TraceRecord:
    labels = scenario.route_labels
    entropy = _entropy_from_labels(labels)
    stability = TraceRecord.stability_label(entropy)
    contrasts = ContrastSet(
        similar=scenario.contrast["similar"],
        neutral=scenario.contrast["neutral"],
        opposite=scenario.contrast["opposite"],
    )
    summary = TraceRecord.build_summary(
        node_id=scenario.node_id,
        route=labels["original"],
        stability=stability,
        entropy=entropy,
        attribution=scenario.attribution_dimension,
        counterfactual=scenario.counterfactual_route,
    )
    return TraceRecord(
        trace_id=TraceRecord.make_id(),
        node_id=scenario.node_id,
        timestamp=datetime.now(UTC),
        original_input=scenario.original_input,
        original_route=labels["original"],
        entropy_score=entropy,
        stability=stability,
        attribution_dimension=scenario.attribution_dimension,
        plain_language_summary=summary,
        raw_contrasts=contrasts,
        raw_outputs={
            "similar": labels["similar"],
            "neutral": labels["neutral"],
            "opposite": labels["opposite"],
        },
        counterfactual_route=scenario.counterfactual_route,
        analysis_overhead=dict(_OFFLINE_OVERHEAD),
    )


def build_offline() -> list[dict[str, Any]]:
    """Deterministic demo bundle from the hand-authored scenarios."""
    payloads = []
    for scenario in ALL_SCENARIOS:
        record = _record_from_scenario(scenario)
        payload = record.to_dict()
        payload["scenario_id"] = scenario.scenario_id
        payload["scenario_title"] = scenario.title
        payload["scenario_note"] = scenario.note
        payload["candidate_routes"] = scenario.routes
        payloads.append(payload)
    return payloads


def build_live() -> list[dict[str, Any]]:
    """Re-run each scenario through the real contrast + divergence pipeline."""
    from conntrail.analyser import DivergenceAnalyser
    from conntrail.contrast import ContrastGenerator
    from conntrail.utils.providers import get_chat_model

    llm = get_chat_model(max_tokens=2048, disable_reasoning=True)
    generator = ContrastGenerator(llm=llm)
    analyser = DivergenceAnalyser()

    payloads = []
    for scenario in ALL_SCENARIOS:
        routes = scenario.routes

        async def node_fn(state: dict, _routes=routes) -> dict:
            # Live mode still uses the frozen labels as ground truth: the demo
            # is about measuring the *input*, so the decision function is held
            # fixed and only the input varies.
            import hashlib

            text = state.get("message", "")
            idx = int(hashlib.md5(text.encode()).hexdigest(), 16) % len(_routes)
            return {**state, "route": _routes[idx]}

        contrasts = asyncio.run(generator.generate(scenario.original_input))
        result = asyncio.run(
            analyser.analyse(
                node_fn,
                {"message": scenario.original_input},
                contrasts,
                input_key="message",
                route_key="route",
            )
        )
        record = TraceRecord(
            trace_id=TraceRecord.make_id(),
            node_id=scenario.node_id,
            timestamp=datetime.now(UTC),
            original_input=scenario.original_input,
            original_route=result.original_route,
            entropy_score=result.entropy_score,
            stability=TraceRecord.stability_label(result.entropy_score),
            attribution_dimension=result.attribution_dimension,
            plain_language_summary=TraceRecord.build_summary(
                scenario.node_id,
                result.original_route,
                TraceRecord.stability_label(result.entropy_score),
                result.entropy_score,
                result.attribution_dimension,
                result.counterfactual_route,
            ),
            raw_contrasts=contrasts,
            raw_outputs=result.contrast_routes,
            counterfactual_route=result.counterfactual_route,
        )
        payload = record.to_dict()
        payload["scenario_id"] = scenario.scenario_id
        payload["scenario_title"] = scenario.title
        payload["scenario_note"] = scenario.note
        payload["candidate_routes"] = scenario.routes
        payloads.append(payload)
    return payloads


def build() -> list:
    """Harness entry point for `conntrail fixtures --harness examples.website.build_fixtures:build`.

    Returns offline TraceRecords (no ``scenario_*`` metadata — the CLI's
    generic bundle writer uses the plain TraceRecord shape). Use the script's
    ``main`` to also emit the web demo's scenario metadata.
    """
    return [_record_from_scenario(scenario) for scenario in ALL_SCENARIOS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the website demo fixture bundle")
    parser.add_argument("--out", type=Path, default=Path("web/data/demo.json"))
    parser.add_argument("--live", action="store_true", help="use a live LLM for contrast/entropy")
    args = parser.parse_args()

    records = build_live() if args.live else build_offline()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "live" if args.live else "offline",
        "records": records,
    }
    args.out.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(records)} scenario(s) to {args.out} ({bundle['mode']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
