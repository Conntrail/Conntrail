"""
dspy.Module student for the customer-support routing decision (G1).

Categories and prompt intent are adapted from Conntrail-Lib's
experiments/agents/customer_support/adapter.py as reference — not imported.
This is a new, independent dspy.Module built for Phase 5's live GEPA run
(G2); none of Conntrail-Lib's LangGraph-based agents are dspy modules, so
this exists purely as the optimizer's student.
"""
from __future__ import annotations

import dspy

VALID_CATEGORIES = ("refund", "escalation", "order_info", "general")


class ClassifyCustomerQuery(dspy.Signature):
    """Classify a customer support message into exactly one category:
    refund (wants a refund, return, or money back), escalation (angry, wants
    a manager, or the situation is urgent/critical), order_info (asking
    about order status, tracking, or shipping), or general (anything else).
    """

    message: str = dspy.InputField(desc="the customer's support message")
    category: str = dspy.OutputField(desc="one of: refund, escalation, order_info, general")


class CustomerSupportRouter(dspy.Module):
    """Routes a customer support message to one of four categories.

    Standalone-runnable (see acceptance criteria for G1): call it directly
    with a message to get a category prediction, independent of GEPA.
    """

    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.Predict(ClassifyCustomerQuery)

    def forward(self, message: str) -> dspy.Prediction:
        result = self.classify(message=message)
        category = (result.category or "").strip().lower()
        if category not in VALID_CATEGORIES:
            category = "general"
        return dspy.Prediction(category=category)
