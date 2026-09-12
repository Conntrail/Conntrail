"""
Tests for conntrail.cost_analyzer — every finding dimension, firing and
not firing. Uses plain TraceRecords with hand-built cost telemetry dicts.
"""
from __future__ import annotations

from datetime import UTC, datetime

from conntrail.contrast import ContrastSet
from conntrail.cost_analyzer import analyze_cost
from conntrail.record import TraceRecord


def _make_record(
    *,
    token_usage: dict | None = None,
    analysis_overhead: dict | None = None,
    cost_usd: float | None = None,
    status: str = "ok",
    error_type: str | None = None,
    route: str = "refund",
) -> TraceRecord:
    return TraceRecord(
        trace_id="t1",
        node_id="router",
        timestamp=datetime.now(UTC),
        original_input="I need a refund",
        original_route=route,
        entropy_score=0.2,
        stability="confident",
        attribution_dimension="urgency/sentiment",
        plain_language_summary="summary",
        raw_contrasts=ContrastSet(similar="s", neutral="n", opposite="o"),
        raw_outputs={},
        counterfactual_route=None,
        status=status,
        error_type=error_type,
        error_message="boom" if error_type else None,
        token_usage=token_usage,
        cost_usd=cost_usd,
        latency_ms=42.0,
        analysis_overhead=analysis_overhead,
    )


def _findings_by_dimension(record) -> dict[str, list]:
    return {
        f["dimension"]: f
        for f in analyze_cost(record)
    }


def _usage(**overrides) -> dict:
    base = {
        "llm_call_count": 3,
        "input_tokens": 4000,
        "output_tokens": 60,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 4060,
        "models": ["claude-haiku-4-5-20251001"],
        "prompt_chars": 16000,
        "system_prompt_chars": 8000,
        "prompt_token_estimate": 4000,
        "prompt_hashes": [],
        "repeated_prompt_blocks": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# cache_efficiency
# ---------------------------------------------------------------------------


class TestCacheEfficiency:
    def test_missing_cache_control_on_anthropic_fires_warning(self):
        record = _make_record(token_usage=_usage(input_tokens=4000, cached_input_tokens=0, cache_write_tokens=0))
        finding = _findings_by_dimension(record)["cache_efficiency"]
        assert finding["severity"] == "warning"
        assert "cache_control" in finding["recommendation"]

    def test_zero_cache_activity_on_non_anthropic_does_not_fire(self):
        record = _make_record(token_usage=_usage(models=["gpt-4o"]))
        assert "cache_efficiency" not in _findings_by_dimension(record)

    def test_below_cacheable_size_does_not_fire(self):
        record = _make_record(token_usage=_usage(input_tokens=500))
        assert "cache_efficiency" not in _findings_by_dimension(record)

    def test_cache_writes_never_read_fires_warning(self):
        record = _make_record(
            token_usage=_usage(cache_write_tokens=2000, cached_input_tokens=0)
        )
        cache_findings = [
            f for f in analyze_cost(record) if f["dimension"] == "cache_efficiency"
        ]
        evidence = " ".join(f["evidence"] for f in cache_findings)
        assert any(f["severity"] == "warning" for f in cache_findings)
        assert "cache-write" in evidence

    def test_low_hit_rate_on_large_prompts_fires_warning(self):
        record = _make_record(
            token_usage=_usage(input_tokens=8000, cached_input_tokens=400)  # 5%
        )
        finding = _findings_by_dimension(record)["cache_efficiency"]
        assert finding["severity"] == "warning"
        assert "5%" in finding["evidence"]

    def test_healthy_hit_rate_fires_info(self):
        record = _make_record(
            token_usage=_usage(input_tokens=4000, cached_input_tokens=3600)  # 90%
        )
        finding = _findings_by_dimension(record)["cache_efficiency"]
        assert finding["severity"] == "info"
        assert "90%" in finding["evidence"]

    def test_no_usage_no_finding(self):
        record = _make_record(token_usage=None)
        assert "cache_efficiency" not in _findings_by_dimension(record)


# ---------------------------------------------------------------------------
# prompt_size
# ---------------------------------------------------------------------------


class TestPromptSize:
    def test_very_large_prompt_fires_warning(self):
        # 30000 tokens over 3 calls = 10000 per call.
        record = _make_record(token_usage=_usage(input_tokens=30000))
        finding = _findings_by_dimension(record)["prompt_size"]
        assert finding["severity"] == "warning"
        assert "~10000" in finding["evidence"]

    def test_static_dominant_and_large_fires_warning(self):
        record = _make_record(
            token_usage=_usage(prompt_chars=20000, system_prompt_chars=18000, prompt_token_estimate=5000)
        )
        finding = _findings_by_dimension(record)["prompt_size"]
        assert finding["severity"] == "warning"
        assert "90%" in finding["evidence"]

    def test_static_dominant_but_small_fires_info(self):
        record = _make_record(
            token_usage=_usage(prompt_chars=4000, system_prompt_chars=3600, prompt_token_estimate=1000)
        )
        finding = _findings_by_dimension(record)["prompt_size"]
        assert finding["severity"] == "info"

    def test_normal_size_no_finding(self):
        record = _make_record(token_usage=_usage())
        assert "prompt_size" not in _findings_by_dimension(record)


# ---------------------------------------------------------------------------
# repeated_instructions
# ---------------------------------------------------------------------------


class TestRepeatedInstructions:
    def test_repeated_blocks_with_low_cache_fires_warning(self):
        record = _make_record(token_usage=_usage(repeated_prompt_blocks=3))
        finding = _findings_by_dimension(record)["repeated_instructions"]
        assert finding["severity"] == "warning"
        assert "cached prefix" in finding["recommendation"]

    def test_repeated_blocks_with_good_cache_fires_info(self):
        record = _make_record(
            token_usage=_usage(input_tokens=4000, cached_input_tokens=3600, repeated_prompt_blocks=3)
        )
        finding = _findings_by_dimension(record)["repeated_instructions"]
        assert finding["severity"] == "info"

    def test_no_repetition_no_finding(self):
        record = _make_record(token_usage=_usage())
        assert "repeated_instructions" not in _findings_by_dimension(record)


# ---------------------------------------------------------------------------
# output_discipline
# ---------------------------------------------------------------------------


class TestOutputDiscipline:
    def test_verbose_output_for_label_route_fires_warning(self):
        record = _make_record(token_usage=_usage(output_tokens=300))  # 100/call
        finding = _findings_by_dimension(record)["output_discipline"]
        assert finding["severity"] == "warning"
        assert "constrained" in finding["recommendation"]

    def test_terse_output_no_finding(self):
        record = _make_record(token_usage=_usage(output_tokens=15))
        assert "output_discipline" not in _findings_by_dimension(record)

    def test_unknown_route_fires_warning(self):
        record = _make_record(route="unknown")
        finding = _findings_by_dimension(record)["output_discipline"]
        assert finding["severity"] == "warning"
        assert "'unknown'" in finding["evidence"]

    def test_retry_loop_error_with_usage_fires_warning(self):
        record = _make_record(
            status="error",
            error_type="retry_loop",
            token_usage=_usage(total_tokens=9000),
        )
        finding = _findings_by_dimension(record)["output_discipline"]
        assert finding["severity"] == "warning"
        assert "9000" in finding["evidence"]

    def test_other_error_no_output_finding(self):
        record = _make_record(status="error", error_type="ValueError")
        assert "output_discipline" not in _findings_by_dimension(record)


# ---------------------------------------------------------------------------
# observer_overhead
# ---------------------------------------------------------------------------


class TestObserverOverhead:
    def test_overhead_present_fires_info(self):
        record = _make_record(
            token_usage=_usage(total_tokens=5000),
            analysis_overhead={"total_tokens": 2000, "cost_usd": 0.005, "retries": 0, "latency_ms": 100.0},
        )
        finding = _findings_by_dimension(record)["observer_overhead"]
        assert finding["severity"] == "info"
        assert "2000" in finding["evidence"]

    def test_overhead_dominating_node_fires_warning(self):
        record = _make_record(
            token_usage=_usage(total_tokens=1000),
            analysis_overhead={"total_tokens": 8000, "cost_usd": 0.02, "retries": 1, "latency_ms": 300.0},
        )
        finding = _findings_by_dimension(record)["observer_overhead"]
        assert finding["severity"] == "warning"
        assert "8.0x" in finding["evidence"]

    def test_zero_overhead_no_finding(self):
        record = _make_record(
            analysis_overhead={"total_tokens": 0, "cost_usd": 0.0, "retries": 0}
        )
        assert "observer_overhead" not in _findings_by_dimension(record)


# ---------------------------------------------------------------------------
# overall behavior
# ---------------------------------------------------------------------------


def test_findings_sorted_in_dimension_order():
    record = _make_record(
        token_usage=_usage(
            input_tokens=30000,
            repeated_prompt_blocks=2,
            output_tokens=300,
        ),
        analysis_overhead={"total_tokens": 100000, "cost_usd": 0.5, "retries": 0},
    )
    dims = [f["dimension"] for f in analyze_cost(record)]
    assert dims == sorted(
        dims,
        key=["cache_efficiency", "prompt_size", "repeated_instructions", "output_discipline", "observer_overhead"].index,
    )


def test_plain_legacy_record_yields_no_findings():
    assert analyze_cost(_make_record()) == []
