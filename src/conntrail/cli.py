"""
Conntrail command-line interface.

Subcommands:
    fixtures   Freeze TraceRecords to a JSON bundle via a ``module:callable``
               harness (used to build the static website demo).
    gate       Fail (non-zero exit) when a captured fixture bundle contains a
               routing decision above an entropy threshold — a CI primitive.
    report     Generate a report from collector data (not yet implemented).

The CLI is deliberately dependency-light so it can run inside CI and in the
static-site build without pulling the server or dashboard extras.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from conntrail.fixtures import load_fixture_bundle, load_harness, write_fixture_bundle
from conntrail.record import TraceRecord

_STABILITY_LABELS = ("confident", "boundary", "fragile")


def _cmd_fixtures(args: argparse.Namespace) -> int:
    harness = load_harness(args.harness)
    records = harness()
    if not isinstance(records, list):
        raise TypeError(
            f"harness {args.harness!r} must return a list[TraceRecord], "
            f"got {type(records).__name__}"
        )
    for record in records:
        if not isinstance(record, TraceRecord):
            raise TypeError(
                f"harness {args.harness!r} returned a non-TraceRecord: {type(record).__name__}"
            )
    out = write_fixture_bundle(records, args.out)
    print(f"wrote {len(records)} record(s) to {out}")
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    records = load_fixture_bundle(args.input)
    if args.stability:
        records = [r for r in records if r.get("stability") in args.stability]

    breaches = []
    for record in records:
        entropy = float(record.get("entropy_score") or 0.0)
        if entropy >= args.threshold:
            breaches.append(
                {
                    "node_id": record.get("node_id"),
                    "route": record.get("original_route"),
                    "entropy_score": entropy,
                    "stability": record.get("stability"),
                    "attribution_dimension": record.get("attribution_dimension"),
                    "counterfactual_route": record.get("counterfactual_route"),
                }
            )

    summary = {
        "input": str(args.input),
        "threshold": args.threshold,
        "records_checked": len(records),
        "breaches": breaches,
        "passed": not breaches,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"checked {len(records)} record(s) at threshold {args.threshold}: "
            f"{len(breaches)} breach(es)"
        )
        for breach in breaches:
            print(
                f"  - {breach['node_id']}: route={breach['route']} "
                f"entropy={breach['entropy_score']:.3f} ({breach['stability']}), "
                f"attribution={breach['attribution_dimension']}"
            )
    return 1 if breaches else 0


def _cmd_report(args: argparse.Namespace) -> int:
    if args.kind != "cost":
        print(
            f"conntrail report --kind {args.kind} is not implemented yet.",
            file=sys.stderr,
        )
        return 2

    from conntrail_client import CollectorClient
    from conntrail_reports import audit_data_from_bundle, build_audit_data, render_html, render_pdf

    if args.bundle is not None:
        import json

        bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
        # Accept either a cost-audit bundle ({"cost_summary","traces"}) or a
        # plain fixture bundle ({"records": [...]}).
        if "cost_summary" not in bundle and "records" in bundle:
            bundle = {"cost_summary": _summarize_records(bundle["records"]), "traces": bundle["records"]}
        data = audit_data_from_bundle(bundle)
    else:
        import asyncio

        client = CollectorClient(base_url=args.collector_url, api_key=args.api_key)
        bundle = asyncio.run(client.get_export(node_id=args.node_id, since=args.since, until=args.until))
        data = build_audit_data(
            cost_summary=bundle["cost_summary"],
            traces=bundle["traces"],
            source=f"collector:{args.collector_url or 'COLLECTOR_URL'}",
            generated_at=bundle.get("generated_at"),
        )

    out = Path(args.out)
    html = render_html(data)
    if out.suffix.lower() == ".pdf":
        try:
            render_pdf(data, out)
        except ImportError:
            out = out.with_suffix(".html")
            out.write_text(html, encoding="utf-8")
            print(
                "WeasyPrint not installed — wrote HTML instead; open it and "
                "print to PDF from the browser.",
                file=sys.stderr,
            )
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
    print(f"wrote {args.kind} report to {out}")
    return 0


def _summarize_records(records: list[dict]) -> dict:
    """Build a minimal cost_summary from raw records (offline fallback)."""
    nodes: dict[str, dict] = {}
    shared: dict[str, set[str]] = {}
    for record in records:
        node_id = str(record.get("node_id", "unknown"))
        node = nodes.setdefault(
            node_id,
            {
                "node_id": node_id,
                "trace_count": 0,
                "llm_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_tokens": 0,
                "total_cost_usd": 0.0,
                "mean_cost_usd": None,
                "mean_latency_ms": None,
                "cache_hit_ratio": None,
                "cost_warning_count": 0,
            },
        )
        node["trace_count"] += 1
        usage = record.get("token_usage") or {}
        if usage:
            node["llm_call_count"] += int(usage.get("llm_call_count") or 0)
            node["input_tokens"] += int(usage.get("input_tokens") or 0)
            node["output_tokens"] += int(usage.get("output_tokens") or 0)
            node["cached_input_tokens"] += int(usage.get("cached_input_tokens") or 0)
            node["cache_write_tokens"] += int(usage.get("cache_write_tokens") or 0)
            for token_hash in usage.get("prompt_hashes") or []:
                shared.setdefault(token_hash, set()).add(node_id)
        if record.get("cost_usd") is not None:
            node["total_cost_usd"] += float(record["cost_usd"])
        node["cost_warning_count"] += sum(
            1 for f in (record.get("cost_findings") or []) if f.get("severity") == "warning"
        )
    for node in nodes.values():
        if node["input_tokens"]:
            node["cache_hit_ratio"] = round(node["cached_input_tokens"] / node["input_tokens"], 3)
        node["total_cost_usd"] = round(node["total_cost_usd"], 6)
    blocks = [
        {"hash": h, "node_ids": sorted(ids), "occurrences": len(ids)}
        for h, ids in shared.items()
        if len(ids) >= 2
    ]
    blocks.sort(key=lambda b: b["occurrences"], reverse=True)
    return {"nodes": list(nodes.values()), "shared_prompt_blocks": blocks[:20]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conntrail", description="Conntrail CLI")
    parser.add_argument("--version", action="version", version="conntrail 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    fixtures = sub.add_parser("fixtures", help="freeze TraceRecords to a JSON bundle")
    fixtures.add_argument(
        "--harness",
        required=True,
        help="module:callable returning list[TraceRecord] (e.g. examples.website.build_fixtures:build)",
    )
    fixtures.add_argument("--out", required=True, type=Path, help="output JSON bundle path")
    fixtures.set_defaults(func=_cmd_fixtures)

    gate = sub.add_parser("gate", help="fail when a fixture bundle breaches an entropy threshold")
    gate.add_argument("--input", required=True, type=Path, help="fixture bundle JSON path")
    gate.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="entropy threshold; records at or above it fail the gate (default: 0.6)",
    )
    gate.add_argument(
        "--stability",
        nargs="+",
        choices=_STABILITY_LABELS,
        help="only consider records with these stability labels",
    )
    gate.add_argument("--json", action="store_true", help="emit a machine-readable summary")
    gate.set_defaults(func=_cmd_gate)

    report = sub.add_parser("report", help="generate a report (cost audit)")
    report.add_argument("--kind", choices=("cost", "ai-act"), default="cost")
    report.add_argument("--collector-url", default=None)
    report.add_argument("--api-key", default=None)
    report.add_argument("--node-id", default=None)
    report.add_argument("--since", default=None)
    report.add_argument("--until", default=None)
    report.add_argument("--bundle", type=Path, default=None, help="offline fixture bundle")
    report.add_argument("--out", type=Path, default=Path("conntrail-report.html"))
    report.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
