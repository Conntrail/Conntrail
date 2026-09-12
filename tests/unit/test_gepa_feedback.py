"""Unit tests for CPEFeedbackFunction and cpe_feedback — no API keys required."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from conntrail.contrast import ContrastSet
from conntrail.gepa.feedback import CPEFeedbackFunction, cpe_feedback
from conntrail.gepa.schema import PromptAttemptRecord
from conntrail.record import TraceRecord


def _make_trace(
    entropy: float,
    stability: str,
    attribution: str = "urgency/sentiment",
) -> TraceRecord:
    return TraceRecord(
        trace_id="test-id",
        node_id="router",
        timestamp=datetime.now(UTC),
        original_input="test input",
        original_route="route_a",
        entropy_score=entropy,
        stability=stability,
        attribution_dimension=attribution,
        plain_language_summary="test summary",
        raw_contrasts=ContrastSet(similar="s", neutral="n", opposite="o"),
        raw_outputs={},
        counterfactual_route=None,
    )


def _make_attempt(traces: list[TraceRecord], score: float | None = None) -> PromptAttemptRecord:
    attempt = PromptAttemptRecord(attempt_id="a1", prompt_candidate="test prompt")
    attempt.traces = traces
    attempt.scalar_score = score
    return attempt


# --- cpe_feedback ---

def test_feedback_no_traces():
    attempt = _make_attempt([])
    result = cpe_feedback(attempt)
    assert "no conntrail traces" in result.lower()


def test_feedback_dominant_fragile():
    traces = [_make_trace(0.9, "fragile")] * 3 + [_make_trace(0.1, "confident")]
    result = cpe_feedback(_make_attempt(traces))
    assert "underspecified" in result or "fragile" in result.lower()


def test_feedback_dominant_boundary():
    traces = [_make_trace(0.4, "boundary")] * 3 + [_make_trace(0.1, "confident")]
    result = cpe_feedback(_make_attempt(traces))
    assert "boundary" in result.lower() or "inconsistent" in result.lower()


def test_feedback_dominant_confident():
    traces = [_make_trace(0.1, "confident")] * 4
    result = cpe_feedback(_make_attempt(traces))
    assert "stable" in result.lower() or "confident" in result.lower()


def test_feedback_includes_attribution():
    traces = [_make_trace(0.8, "fragile", "surface form")] * 2
    result = cpe_feedback(_make_attempt(traces))
    assert "surface form" in result


def test_feedback_includes_task_score_when_not_confident():
    traces = [_make_trace(0.7, "fragile")] * 2
    result = cpe_feedback(_make_attempt(traces, score=0.42))
    assert "0.420" in result or "task score" in result.lower()


def test_feedback_no_task_score_line_when_confident():
    traces = [_make_trace(0.1, "confident")] * 3
    result = cpe_feedback(_make_attempt(traces, score=0.99))
    assert "task score" not in result.lower()


def test_feedback_header_contains_trace_count():
    traces = [_make_trace(0.5, "boundary")] * 7
    result = cpe_feedback(_make_attempt(traces))
    assert "7" in result


# --- PromptAttemptRecord properties ---

def test_mean_entropy():
    attempt = _make_attempt([_make_trace(0.2, "confident"), _make_trace(0.8, "fragile")])
    assert abs(attempt.mean_entropy - 0.5) < 1e-9


def test_mean_entropy_none_when_no_traces():
    assert _make_attempt([]).mean_entropy is None


def test_fragile_count():
    traces = [
        _make_trace(0.9, "fragile"),
        _make_trace(0.1, "confident"),
        _make_trace(0.8, "fragile"),
    ]
    assert _make_attempt(traces).fragile_count == 2


def test_boundary_count():
    traces = [_make_trace(0.4, "boundary")] * 3 + [_make_trace(0.1, "confident")]
    assert _make_attempt(traces).boundary_count == 3


def test_dominant_attribution():
    traces = [
        _make_trace(0.8, "fragile", "semantic intensity"),
        _make_trace(0.7, "fragile", "urgency/sentiment"),
        _make_trace(0.9, "fragile", "semantic intensity"),
    ]
    assert _make_attempt(traces).dominant_attribution == "semantic intensity"


def test_dominant_attribution_none_when_no_traces():
    assert _make_attempt([]).dominant_attribution is None


# --- CPEFeedbackFunction ---

class _FakeCollector:
    def __init__(self, attempts):
        self._attempts = attempts

    @property
    def all_attempts(self):
        return self._attempts


def test_feedback_fn_no_attempts():
    fn = CPEFeedbackFunction(_FakeCollector([]))
    result = fn({}, {}, None, None, None)
    assert result.score == 0.0
    assert "no attempts" in result.feedback.lower()


def test_feedback_fn_uses_task_metric():
    attempt = _make_attempt([_make_trace(0.5, "boundary")], score=None)
    collector = _FakeCollector([attempt])
    fn = CPEFeedbackFunction(collector, task_metric_fn=lambda g, p: 0.75)
    result = fn({}, {}, None, None, None)
    assert result.score == 0.75


def test_feedback_fn_falls_back_to_scalar_score():
    attempt = _make_attempt([_make_trace(0.3, "boundary")], score=0.6)
    collector = _FakeCollector([attempt])
    fn = CPEFeedbackFunction(collector)
    result = fn({}, {}, None, None, None)
    assert result.score == 0.6


def test_feedback_fn_falls_back_to_stability_proxy():
    attempt = _make_attempt([_make_trace(0.4, "boundary")])  # no scalar_score
    collector = _FakeCollector([attempt])
    fn = CPEFeedbackFunction(collector)
    result = fn({}, {}, None, None, None)
    assert abs(result.score - (1.0 - 0.4)) < 1e-9


def test_feedback_fn_returns_cpe_feedback_string():
    traces = [_make_trace(0.9, "fragile")] * 3
    attempt = _make_attempt(traces, score=0.5)
    collector = _FakeCollector([attempt])
    fn = CPEFeedbackFunction(collector)
    result = fn({}, {}, None, None, None)
    assert "CPE Analysis" in result.feedback


def test_feedback_fn_persists_task_metric_score_onto_attempt():
    """end_attempt() has no score to give the record (the caller doesn't have
    gold/pred yet) — the metric call is the first point a real task score
    exists, so it must be written back for later reporting."""
    attempt = _make_attempt([_make_trace(0.5, "boundary")], score=None)
    collector = _FakeCollector([attempt])
    fn = CPEFeedbackFunction(collector, task_metric_fn=lambda g, p: 0.75)
    fn({}, {}, None, None, None)
    assert attempt.scalar_score == 0.75


def test_feedback_fn_fires_on_attempt_scored_hook_with_final_record():
    attempt = _make_attempt([_make_trace(0.5, "boundary")], score=None)
    collector = _FakeCollector([attempt])
    received = []
    fn = CPEFeedbackFunction(
        collector, task_metric_fn=lambda g, p: 0.75, on_attempt_scored=received.append
    )
    fn({}, {}, None, None, None)
    assert received == [attempt]
    assert received[0].scalar_score == 0.75


def test_feedback_fn_does_not_fire_hook_without_task_metric_fn():
    attempt = _make_attempt([_make_trace(0.5, "boundary")], score=0.6)
    collector = _FakeCollector([attempt])
    received = []
    fn = CPEFeedbackFunction(collector, on_attempt_scored=received.append)
    fn({}, {}, None, None, None)
    assert received == []


def test_feedback_fn_satisfies_gepa_metric_signature():
    """dspy 3.x's GEPA constructs metrics via inspect.signature(...).bind(5
    positional args) — the callable must accept exactly that shape."""
    import inspect

    fn = CPEFeedbackFunction(_FakeCollector([]))
    inspect.signature(fn).bind(None, None, None, None, None)


# --- cost telemetry ---

def _make_cost_attempt(
    traces: list[TraceRecord],
    *,
    token_usage: dict | None = None,
    cost_usd: float | None = None,
) -> PromptAttemptRecord:
    attempt = PromptAttemptRecord(
        attempt_id=f"a-{id(traces)}",
        prompt_candidate="test prompt",
        token_usage=token_usage,
        cost_usd=cost_usd,
    )
    attempt.traces = traces
    return attempt


class TestAttemptCostProperties:
    def test_attempt_level_usage_preferred_over_traces(self):
        trace = _make_trace(0.1, "confident")
        trace.token_usage = {"input_tokens": 999, "output_tokens": 999}
        attempt = _make_cost_attempt(
            [trace], token_usage={"input_tokens": 100, "output_tokens": 10}
        )
        assert attempt.total_input_tokens == 100
        assert attempt.total_output_tokens == 10

    def test_trace_derived_fallback(self):
        t1, t2 = _make_trace(0.1, "confident"), _make_trace(0.2, "confident")
        t1.token_usage = {"input_tokens": 100, "output_tokens": 10}
        t2.token_usage = {"input_tokens": 200, "output_tokens": 20}
        attempt = _make_cost_attempt([t1, t2])
        assert attempt.total_input_tokens == 300
        assert attempt.total_output_tokens == 30

    def test_no_data_anywhere_is_none(self):
        attempt = _make_cost_attempt([_make_trace(0.1, "confident")])
        assert attempt.total_input_tokens is None
        assert attempt.total_output_tokens is None
        assert attempt.total_cost_usd is None
        assert attempt.mean_latency_ms is None

    def test_cost_prefers_attempt_level_then_sums_traces(self):
        t1 = _make_trace(0.1, "confident")
        t1.cost_usd = 0.004
        t2 = _make_trace(0.1, "confident")
        t2.cost_usd = 0.001
        assert _make_cost_attempt([t1, t2]).total_cost_usd == pytest.approx(0.005)
        assert _make_cost_attempt([t1], cost_usd=0.05).total_cost_usd == 0.05

    def test_mean_latency_from_traces(self):
        t1, t2 = _make_trace(0.1, "confident"), _make_trace(0.1, "confident")
        t1.latency_ms = 100.0
        t2.latency_ms = 200.0
        assert _make_cost_attempt([t1, t2]).mean_latency_ms == 150.0


class TestCostFoldedScoring:
    def _confident_trace(self):
        return _make_trace(0.1, "confident")

    def test_expensive_candidate_penalized(self):
        baseline = _make_cost_attempt(
            [self._confident_trace()], token_usage={"input_tokens": 100, "output_tokens": 0}
        )
        latest = _make_cost_attempt(
            [self._confident_trace()], token_usage={"input_tokens": 300, "output_tokens": 0}
        )
        fn = CPEFeedbackFunction(
            _FakeCollector([baseline, latest]), task_metric_fn=lambda g, p: 0.8
        )
        result = fn({}, {}, None, None, None)
        # ratio 3.0 → 0.8 - 0.1 * (3.0 - 1.0)
        assert result.score == pytest.approx(0.6)

    def test_scalar_score_stays_raw_task_score(self):
        baseline = _make_cost_attempt(
            [self._confident_trace()], token_usage={"input_tokens": 100, "output_tokens": 0}
        )
        latest = _make_cost_attempt(
            [self._confident_trace()], token_usage={"input_tokens": 300, "output_tokens": 0}
        )
        fn = CPEFeedbackFunction(
            _FakeCollector([baseline, latest]), task_metric_fn=lambda g, p: 0.8
        )
        result = fn({}, {}, None, None, None)
        assert result.score == pytest.approx(0.6)
        assert latest.scalar_score == 0.8  # raw accuracy, not cost-adjusted

    def test_cheaper_candidate_rewarded(self):
        baseline = _make_cost_attempt(
            [self._confident_trace()], token_usage={"input_tokens": 200, "output_tokens": 0}
        )
        latest = _make_cost_attempt(
            [self._confident_trace()], token_usage={"input_tokens": 100, "output_tokens": 0}
        )
        fn = CPEFeedbackFunction(
            _FakeCollector([baseline, latest]), task_metric_fn=lambda g, p: 0.8
        )
        result = fn({}, {}, None, None, None)
        # ratio 0.5 → 0.8 - 0.1 * (0.5 - 1.0) = 0.85
        assert result.score == pytest.approx(0.85)

    def test_first_attempt_unpenalized(self):
        only = _make_cost_attempt(
            [self._confident_trace()], token_usage={"input_tokens": 5000, "output_tokens": 500}
        )
        fn = CPEFeedbackFunction(_FakeCollector([only]), task_metric_fn=lambda g, p: 0.7)
        assert fn({}, {}, None, None, None).score == pytest.approx(0.7)

    def test_no_usage_data_means_no_adjustment(self):
        baseline = _make_cost_attempt([self._confident_trace()])
        latest = _make_cost_attempt([self._confident_trace()])
        fn = CPEFeedbackFunction(
            _FakeCollector([baseline, latest]), task_metric_fn=lambda g, p: 0.75
        )
        assert fn({}, {}, None, None, None).score == pytest.approx(0.75)

    def test_zero_cost_weight_disables_folding(self):
        baseline = _make_cost_attempt(
            [self._confident_trace()], token_usage={"input_tokens": 100, "output_tokens": 0}
        )
        latest = _make_cost_attempt(
            [self._confident_trace()], token_usage={"input_tokens": 900, "output_tokens": 0}
        )
        fn = CPEFeedbackFunction(
            _FakeCollector([baseline, latest]),
            task_metric_fn=lambda g, p: 0.8,
            cost_weight=0.0,
        )
        assert fn({}, {}, None, None, None).score == pytest.approx(0.8)

    def test_negative_cost_weight_raises(self):
        with pytest.raises(ValueError, match="cost_weight"):
            CPEFeedbackFunction(_FakeCollector([]), cost_weight=-0.1)

    def test_objective_scores_emitted_for_pareto_tracking(self):
        only = _make_cost_attempt(
            [self._confident_trace()], token_usage={"input_tokens": 300, "output_tokens": 100}
        )
        fn = CPEFeedbackFunction(_FakeCollector([only]), task_metric_fn=lambda g, p: 1.0)
        result = fn({}, {}, None, None, None)
        assert result.objective_scores == {"cost": -400.0}

    def test_objective_scores_absent_without_usage(self):
        only = _make_cost_attempt([self._confident_trace()])
        fn = CPEFeedbackFunction(_FakeCollector([only]), task_metric_fn=lambda g, p: 1.0)
        result = fn({}, {}, None, None, None)
        assert not hasattr(result, "objective_scores")


class TestCostFeedbackText:
    def test_includes_cost_lines_when_usage_present(self):
        attempt = _make_cost_attempt(
            [_make_trace(0.1, "confident")],
            token_usage={"input_tokens": 120, "output_tokens": 30},
            cost_usd=0.001,
        )
        result = cpe_feedback(attempt)
        assert "Cost:" in result
        assert "120 in / 30 out" in result
        assert "$0.0010" in result
        assert "Cost guidance" in result

    def test_omits_cost_lines_without_usage(self):
        result = cpe_feedback(_make_attempt([_make_trace(0.1, "confident")]))
        assert "Cost:" not in result
        assert "Cost guidance" not in result

    def test_flags_costlier_than_baseline(self):
        baseline = _make_cost_attempt(
            [_make_trace(0.1, "confident")], token_usage={"input_tokens": 100, "output_tokens": 0}
        )
        latest = _make_cost_attempt(
            [_make_trace(0.1, "confident")], token_usage={"input_tokens": 400, "output_tokens": 0}
        )
        result = cpe_feedback(latest, baseline=baseline)
        assert "more tokens than the original prompt" in result

    def test_does_not_flag_baseline_itself(self):
        only = _make_cost_attempt(
            [_make_trace(0.1, "confident")], token_usage={"input_tokens": 400, "output_tokens": 0}
        )
        result = cpe_feedback(only, baseline=only)
        assert "more tokens than the original prompt" not in result
