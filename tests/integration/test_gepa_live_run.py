"""
Integration test: a real (non-mocked) CPEGEPAOptimizer.compile() run (G2).

Marked integration+slow per EPICS G2's test plan — a small compile against a
live LLM, verifying that compile() completes without raising and produces
>=2 PromptAttemptRecords carrying real (non-mocked) TraceRecords.

Model selection honors the same env conventions as the rest of the live-test
rig (see tests/conftest.py and .env.example):
  - CONNTRAIL_CONTRAST_MODEL — contrast generation during tracing
  - CONNTRAIL_GEPA_STUDENT_MODEL — the student LM dspy configures
  - CONNTRAIL_GEPA_REFLECTION_MODEL — GEPA's reflection LM
All default to Anthropic models (the original G2 scope); setting them to
"local/<name>" runs the same path against the local server.
"""
from __future__ import annotations

import os

import dspy
import pytest

from conntrail import ConntrailConfig
from conntrail.gepa import CPEGEPAOptimizer
from examples.gepa.customer_support_student import CustomerSupportRouter
from examples.gepa.run_live import make_lm, make_task_metric_fn, make_traced_router_class
from examples.gepa.trainset import TRAINSET
from tests.conftest import default_contrast_model, live_llm_available


def _model_from_env(env_var: str, default: str) -> str:
    return os.environ.get(env_var) or default


@pytest.mark.integration
@pytest.mark.slow
class TestGEPALiveRun:
    @pytest.fixture(autouse=True)
    def require_live_llm(self):
        if not live_llm_available():
            pytest.skip("No live LLM available (cloud key or local server)")

    @pytest.fixture(autouse=True)
    def _restore_dspy_settings(self):
        """dspy.settings is process-global — save/restore around the test."""
        original_lm = dspy.settings.lm
        yield
        dspy.settings.configure(lm=original_lm)

    def test_compile_produces_real_attempts(self):
        student_model = _model_from_env(
            "CONNTRAIL_GEPA_STUDENT_MODEL", "claude-haiku-4-5-20251001"
        )
        reflection_model = _model_from_env("CONNTRAIL_GEPA_REFLECTION_MODEL", "claude-opus-5")
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        # Small on purpose — this is a correctness run, not a full optimization.
        trainset = TRAINSET[:4]
        dspy.settings.configure(lm=make_lm(student_model, api_key=api_key, max_tokens=100))

        optimizer = CPEGEPAOptimizer(
            student=None,  # replaced with the traced router below (see run_live.py)
            trainset=trainset,
            task_metric_fn=make_task_metric_fn(),
            base_conntrail_config=ConntrailConfig(
                contrast_model=default_contrast_model(),
                sample_rate=1.0,
                entropy_alert_threshold=0.0,
                async_mode=False,
            ),
            gepa_kwargs={
                "max_metric_calls": 6,
                "reflection_lm": make_lm(
                    reflection_model, api_key=api_key, max_tokens=1500
                ),
                # TraceCollector assumes one attempt in flight at a time.
                "num_threads": 1,
            },
        )
        traced_router_cls = make_traced_router_class(
            optimizer.collector, optimizer.conntrail_config
        )
        optimizer.student = traced_router_cls(CustomerSupportRouter())

        optimized = optimizer.compile()  # must complete without raising

        attempts = optimizer.attempt_records
        assert len(attempts) >= 2, f"expected >=2 attempt records, got {len(attempts)}"

        with_traces = [a for a in attempts if a.traces]
        assert with_traces, "no attempt carried a real TraceRecord"
        for attempt in with_traces:
            for trace in attempt.traces:
                assert trace.node_id == "classify_query"
                assert 0.0 <= trace.entropy_score <= 1.0

        mean_entropies = [a.mean_entropy for a in with_traces if a.mean_entropy is not None]
        assert mean_entropies, "no attempt produced a real mean_entropy value"

        # The optimized student must still be a runnable dspy module.
        assert optimized is not None
