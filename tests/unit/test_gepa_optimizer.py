"""Unit tests for CPEGEPAOptimizer — dspy is mocked throughout."""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from conntrail.config import ConntrailConfig
from conntrail.contrast import ContrastSet
from conntrail.gepa.bridge import TraceCollector
from conntrail.gepa.feedback import CPEFeedbackFunction
from conntrail.gepa.optimizer import CPEGEPAOptimizer
from conntrail.record import TraceRecord


class _FakePrediction:
    """Stand-in for dspy.Prediction — attribute access only, no dspy dependency."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


# Inject a fake dspy module so compile() can import it without the package installed
def _make_dspy_mock():
    mod = ModuleType("dspy")
    mod.GEPA = MagicMock()
    mod.Prediction = _FakePrediction
    return mod


@pytest.fixture(autouse=True)
def mock_dspy(monkeypatch):
    fake = _make_dspy_mock()
    monkeypatch.setitem(sys.modules, "dspy", fake)
    return fake


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


def _make_optimizer(**kwargs) -> CPEGEPAOptimizer:
    defaults = dict(
        student=MagicMock(),
        trainset=[{"input": "hi", "expected_route": "a"}],
        base_conntrail_config=ConntrailConfig(
            sample_rate=1.0,
            entropy_alert_threshold=0.0,
            async_mode=False,
        ),
    )
    defaults.update(kwargs)
    return CPEGEPAOptimizer(**defaults)


# --- construction ---

def test_collector_created_on_init():
    opt = _make_optimizer()
    assert isinstance(opt.collector, TraceCollector)


def test_feedback_fn_created_on_init():
    opt = _make_optimizer()
    assert isinstance(opt.feedback_fn, CPEFeedbackFunction)


def test_conntrail_config_has_on_alert():
    opt = _make_optimizer()
    assert opt.conntrail_config.on_alert is not None


def test_conntrail_config_inherits_base_fields():
    base = ConntrailConfig(
        sample_rate=0.5,
        async_mode=False,
        entropy_alert_threshold=0.0,
        timeout_seconds=2.5,
    )
    opt = _make_optimizer(base_conntrail_config=base)
    assert opt.conntrail_config.sample_rate == 0.5
    assert opt.conntrail_config.async_mode is False
    assert opt.conntrail_config.timeout_seconds == 2.5


def test_task_metric_wired_into_feedback_fn():
    metric = lambda g, p: 0.99  # noqa: E731
    opt = _make_optimizer(task_metric_fn=metric)
    # Seed a completed attempt so the feedback fn has data
    opt.collector.begin_attempt("p")
    opt.collector._current.traces.append(_make_trace())
    opt.collector.end_attempt(scalar_score=None)

    result = opt.feedback_fn({}, {}, None, None, None)
    assert result.score == 0.99


# --- compile() ---

def test_compile_calls_dspy_gepa(mock_dspy):
    mock_gepa_instance = MagicMock()
    mock_dspy.GEPA.return_value = mock_gepa_instance
    mock_gepa_instance.compile.return_value = MagicMock(name="optimized")

    opt = _make_optimizer(gepa_kwargs={"num_iterations": 5})
    result = opt.compile()

    mock_dspy.GEPA.assert_called_once_with(
        metric=opt.feedback_fn,
        num_iterations=5,
    )
    mock_gepa_instance.compile.assert_called_once_with(opt.student, trainset=opt.trainset)
    assert result is mock_gepa_instance.compile.return_value


def test_compile_with_no_gepa_kwargs(mock_dspy):
    mock_gepa_instance = MagicMock()
    mock_dspy.GEPA.return_value = mock_gepa_instance

    opt = _make_optimizer()
    opt.compile()

    mock_dspy.GEPA.assert_called_once_with(metric=opt.feedback_fn)


# --- attempt_records ---

def test_attempt_records_empty_before_compile():
    opt = _make_optimizer()
    assert opt.attempt_records == []


def test_attempt_records_reflects_collector():
    opt = _make_optimizer()
    opt.collector.begin_attempt("p1")
    opt.collector._current.traces.append(_make_trace(0.8, "fragile"))
    opt.collector.end_attempt(scalar_score=0.7)

    records = opt.attempt_records
    assert len(records) == 1
    assert records[0].prompt_candidate == "p1"
    assert records[0].scalar_score == 0.7
    assert records[0].fragile_count == 1
