"""Unit tests for CPEFeedbackFunction and cpe_feedback — no API keys required."""
from __future__ import annotations

from datetime import UTC, datetime

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
