"""
Tests for the conntrail CLI (fixtures, gate, report stub).
"""
from __future__ import annotations

import json

import pytest

from conntrail import cli
from conntrail.fixtures import write_fixture_bundle
from conntrail.record import TraceRecord
from tests.fixtures.trace_records import make_trace_payload


def _record(node_id: str = "router", entropy: float = 0.1) -> TraceRecord:
    return TraceRecord.from_dict(
        make_trace_payload(
            node_id=node_id,
            entropy_score=entropy,
            stability=TraceRecord.stability_label(entropy),
        )
    )


def _harness_ok() -> list[TraceRecord]:
    return [_record("router", 0.1), _record("classifier", 0.8)]


def _harness_bad() -> list:
    return ["not-a-record"]


def _bundle(tmp_path):
    out = tmp_path / "bundle.json"
    write_fixture_bundle([_record("router", 0.1), _record("classifier", 0.8)], out)
    return out


def test_fixtures_command_writes_bundle(tmp_path):
    out = tmp_path / "bundle.json"
    rc = cli.main(["fixtures", "--harness", "tests.unit.test_cli:_harness_ok", "--out", str(out)])

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 2
    assert payload["generated_at"]


def test_fixtures_command_rejects_non_records(tmp_path):
    with pytest.raises(TypeError):
        cli.main(
            [
                "fixtures",
                "--harness",
                "tests.unit.test_cli:_harness_bad",
                "--out",
                str(tmp_path / "b.json"),
            ]
        )


def test_gate_passes_below_threshold(tmp_path, capsys):
    rc = cli.main(["gate", "--input", str(_bundle(tmp_path)), "--threshold", "0.9"])
    assert rc == 0
    assert "0 breach" in capsys.readouterr().out


def test_gate_fails_above_threshold(tmp_path, capsys):
    rc = cli.main(["gate", "--input", str(_bundle(tmp_path)), "--threshold", "0.6"])
    assert rc == 1
    assert "1 breach" in capsys.readouterr().out


def test_gate_json_summary(tmp_path, capsys):
    rc = cli.main(["gate", "--input", str(_bundle(tmp_path)), "--threshold", "0.6", "--json"])
    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["passed"] is False
    assert summary["records_checked"] == 2
    assert summary["breaches"][0]["node_id"] == "classifier"


def test_gate_stability_filter(tmp_path):
    # classifier is fragile (0.8); filtering to confident skips it -> pass.
    rc = cli.main(
        ["gate", "--input", str(_bundle(tmp_path)), "--threshold", "0.6", "--stability", "confident"]
    )
    assert rc == 0


def test_report_ai_act_is_not_implemented(capsys):
    rc = cli.main(["report", "--kind", "ai-act"])
    assert rc == 2
    assert "not implemented" in capsys.readouterr().err


def test_report_offline_bundle_writes_html(tmp_path):
    bundle = {
        "cost_summary": {
            "nodes": [
                {
                    "node_id": "router",
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
            ],
            "shared_prompt_blocks": [],
            "scanned_traces": 2,
        },
        "traces": [
            {
                "node_id": "router",
                "cost_findings": [
                    {
                        "dimension": "cache_efficiency",
                        "severity": "warning",
                        "evidence": "low hit rate",
                        "recommendation": "enable caching",
                    }
                ],
            }
        ],
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    out = tmp_path / "audit.html"

    rc = cli.main(["report", "--kind", "cost", "--bundle", str(bundle_path), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "LLM cost audit" in html
    assert "cache_efficiency" in html


def test_report_offline_fixture_bundle_fallback(tmp_path):
    # A plain fixture bundle (records only) gets a synthetic cost_summary.
    from conntrail.fixtures import write_fixture_bundle

    write_fixture_bundle([_record("router", 0.1)], tmp_path / "fixtures.json")
    out = tmp_path / "audit.html"
    rc = cli.main(
        [
            "report",
            "--kind",
            "cost",
            "--bundle",
            str(tmp_path / "fixtures.json"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert "LLM cost audit" in out.read_text(encoding="utf-8")
