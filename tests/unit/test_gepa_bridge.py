"""Unit tests for TraceCollector — no API keys required."""
from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

from conntrail.config import ConntrailConfig
from conntrail.contrast import ContrastSet
from conntrail.gepa.bridge import TraceCollector
from conntrail.record import TraceRecord


def _make_trace(entropy: float = 0.5, stability: str = "boundary") -> TraceRecord:
    return TraceRecord(
        trace_id="t1",
        node_id="router",
        timestamp=datetime.now(UTC),
        original_input="hello",
        original_route="route_a",
        entropy_score=entropy,
        stability=stability,
        attribution_dimension="urgency/sentiment",
        plain_language_summary="summary",
        raw_contrasts=ContrastSet(similar="s", neutral="n", opposite="o"),
        raw_outputs={},
        counterfactual_route=None,
    )


# --- begin/end attempt lifecycle ---

def test_begin_end_basic():
    collector = TraceCollector()
    collector.begin_attempt("prompt v1")
    record = collector.end_attempt(scalar_score=0.8)
    assert record.prompt_candidate == "prompt v1"
    assert record.scalar_score == 0.8
    assert record.attempt_id  # non-empty UUID string


def test_end_without_begin_raises():
    collector = TraceCollector()
    with pytest.raises(RuntimeError, match="begin_attempt"):
        collector.end_attempt()


def test_double_end_raises():
    collector = TraceCollector()
    collector.begin_attempt("p1")
    collector.end_attempt()
    with pytest.raises(RuntimeError):
        collector.end_attempt()


def test_multiple_sequential_attempts():
    collector = TraceCollector()
    for i in range(3):
        collector.begin_attempt(f"prompt v{i}")
        collector.end_attempt(scalar_score=float(i) / 3)
    assert len(collector.all_attempts) == 3
    assert collector.all_attempts[2].scalar_score == pytest.approx(2 / 3)


def test_all_attempts_returns_copy():
    collector = TraceCollector()
    collector.begin_attempt("p1")
    collector.end_attempt()
    a = collector.all_attempts
    b = collector.all_attempts
    assert a is not b  # independent lists


# --- trace routing via on_alert ---

def test_make_config_routes_alerts_to_current_attempt():
    collector = TraceCollector()
    config = collector.make_config()
    collector.begin_attempt("p1")
    config.on_alert(_make_trace(0.9, "fragile"))
    config.on_alert(_make_trace(0.7, "fragile"))
    record = collector.end_attempt()
    assert len(record.traces) == 2


def test_make_config_ignores_alert_outside_attempt():
    """Traces fired when no attempt is active are silently dropped."""
    collector = TraceCollector()
    config = collector.make_config()
    config.on_alert(_make_trace())  # no active attempt — should not raise
    assert collector.all_attempts == []


def test_make_config_preserves_original_alert():
    received = []
    base = ConntrailConfig(on_alert=lambda t: received.append(t))
    collector = TraceCollector()
    config = collector.make_config(base)
    collector.begin_attempt("p1")
    t = _make_trace()
    config.on_alert(t)
    collector.end_attempt()
    assert received == [t]


def test_make_config_copies_base_fields():
    base = ConntrailConfig(
        sample_rate=0.5,
        async_mode=False,
        timeout_seconds=2.5,
        entropy_alert_threshold=0.0,
    )
    collector = TraceCollector()
    config = collector.make_config(base)
    assert config.sample_rate == 0.5
    assert config.async_mode is False
    assert config.timeout_seconds == 2.5
    assert config.entropy_alert_threshold == 0.0


def test_make_config_with_no_base_uses_defaults():
    collector = TraceCollector()
    config = collector.make_config()
    default = ConntrailConfig()
    assert config.sample_rate == default.sample_rate
    assert config.async_mode == default.async_mode
    assert config.on_alert is not None  # collector's callback


# --- thread safety ---

def test_thread_safe_trace_collection():
    collector = TraceCollector()
    collector.begin_attempt("threaded")
    config = collector.make_config()

    errors = []

    def fire_alert():
        try:
            config.on_alert(_make_trace())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=fire_alert) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    record = collector.end_attempt()
    assert not errors
    assert len(record.traces) == 20
