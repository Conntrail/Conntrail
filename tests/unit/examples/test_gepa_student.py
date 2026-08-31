"""
Tests for CustomerSupportRouter (G1) — structural correctness of the
dspy.Module itself, independent of GEPA. Uses dspy's DummyLM so no real
LLM call happens; this ticket verifies the module's plumbing, not live
classification accuracy (that's exercised for real in G2).
"""
from __future__ import annotations

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from examples.gepa.customer_support_student import VALID_CATEGORIES, CustomerSupportRouter
from examples.gepa.trainset import TRAINSET


@pytest.fixture(autouse=True)
def _isolated_dspy_settings():
    """dspy.settings is process-global — save/restore around each test."""
    original_lm = dspy.settings.lm
    yield
    dspy.settings.configure(lm=original_lm)


@pytest.mark.parametrize(
    "canned_category",
    ["refund", "escalation", "order_info", "general"],
)
def test_forward_returns_one_of_the_valid_categories(canned_category):
    dspy.settings.configure(lm=DummyLM([{"category": canned_category}]))
    router = CustomerSupportRouter()

    prediction = router(message="I need help with my order")

    assert prediction.category == canned_category
    assert prediction.category in VALID_CATEGORIES


def test_forward_normalises_whitespace_and_case():
    dspy.settings.configure(lm=DummyLM([{"category": "  REFUND \n"}]))
    router = CustomerSupportRouter()

    prediction = router(message="give me my money back")

    assert prediction.category == "refund"


def test_forward_falls_back_to_general_on_invalid_category():
    dspy.settings.configure(lm=DummyLM([{"category": "not_a_real_category"}]))
    router = CustomerSupportRouter()

    prediction = router(message="asdf")

    assert prediction.category == "general"


def test_forward_produces_valid_category_for_each_trainset_example():
    """Run the module standalone over a handful of fixture inputs (G1's
    acceptance criteria: it must work independent of being handed to GEPA)."""
    sample = TRAINSET[:5]
    answers = [{"category": ex.category} for ex in sample]
    dspy.settings.configure(lm=DummyLM(answers))
    router = CustomerSupportRouter()

    for example in sample:
        prediction = router(message=example.message)
        assert prediction.category in VALID_CATEGORIES


def test_trainset_examples_have_message_as_input_and_a_valid_expected_category():
    assert len(TRAINSET) >= 5
    for example in TRAINSET:
        assert example.inputs().toDict().keys() == {"message"}
        assert example.category in VALID_CATEGORIES
