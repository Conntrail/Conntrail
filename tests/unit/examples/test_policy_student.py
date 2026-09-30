"""
Tests for the policy-following support resolver and its trainset — structural
correctness only (dspy's DummyLM; no real LLM calls). Live optimization behavior
is exercised by the GEPA run itself.
"""
from __future__ import annotations

from collections import Counter

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from examples.gepa.policy_student import VALID_RESOLUTIONS, PolicyRouter
from examples.gepa.policy_trainset import POLICY_TRAINSET


@pytest.fixture(autouse=True)
def _isolated_dspy_settings():
    original_lm = dspy.settings.lm
    yield
    dspy.settings.configure(lm=original_lm)


@pytest.mark.parametrize("canned", ["refund", "escalate", "no_action", "return_label"])
def test_forward_returns_valid_resolution(canned):
    dspy.settings.configure(lm=DummyLM([{"resolution": canned}]))
    router = PolicyRouter()
    assert router(case="...").resolution == canned


def test_forward_normalises_case_and_spaces():
    dspy.settings.configure(lm=DummyLM([{"resolution": "  Store Credit \n"}]))
    assert PolicyRouter()(case="...").resolution == "store_credit"


def test_forward_falls_back_to_no_action_on_invalid_label():
    dspy.settings.configure(lm=DummyLM([{"resolution": "banana"}]))
    assert PolicyRouter()(case="...").resolution == "no_action"


def test_default_instructions_state_the_policy():
    router = PolicyRouter()
    instructions = router.resolve.signature.instructions.lower()
    # The rules that matter are present by default...
    assert "escalate" in instructions
    assert "30 days" in instructions
    assert "14 days" in instructions
    assert "10 days" in instructions


def test_custom_instructions_override_the_policy():
    router = PolicyRouter(instructions="Just pick something.")
    assert router.resolve.signature.instructions == "Just pick something."
    assert "days" not in router.resolve.signature.instructions


def test_trainset_is_valid_balanced_and_unique():
    assert len(POLICY_TRAINSET) >= 40
    cases = [ex.case for ex in POLICY_TRAINSET]
    assert len(set(cases)) == len(cases), "duplicate cases in trainset"
    for example in POLICY_TRAINSET:
        assert example.inputs().toDict().keys() == {"case"}
        assert example.resolution in VALID_RESOLUTIONS
    counts = Counter(ex.resolution for ex in POLICY_TRAINSET)
    assert set(counts) == set(VALID_RESOLUTIONS)
    assert min(counts.values()) >= 6, f"action imbalance: {dict(counts)}"
