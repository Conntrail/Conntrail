"""
dspy.Module student for a customer-support *policy-following* decision.

Unlike the four-way category classifier (examples/gepa/customer_support_student.py),
which DeepSeek-class models saturate regardless of the prompt, this task can
only be solved by applying a specific, non-obvious policy — so instruction
quality materially moves accuracy and GEPA has real headroom.

The case text states the facts (delivery age, damaged/opened, digital, tone);
the resolution must follow the policy in the signature below.
"""
from __future__ import annotations

import dspy

VALID_RESOLUTIONS = (
    "refund",
    "replacement",
    "store_credit",
    "return_label",
    "escalate",
    "no_action",
)


class ResolveSupportCase(dspy.Signature):
    """Resolve a customer support case into exactly one action, following policy.

    Apply these rules, in order:

    1. escalate — the customer is abusive/threatening, demands a manager, or
       mentions legal action or a regulator. This takes precedence over all
       other rules.
    2. refund — an order that was never delivered and is MORE THAN 10 days past
       its expected delivery date.
    3. replacement — a damaged or defective item that was delivered WITHIN THE
       LAST 30 DAYS.
    4. return_label — a change-of-mind return of an UNOPENED item within 14 days
       of delivery.
    5. store_credit — a damaged item delivered MORE THAN 30 DAYS ago, or a
       change-of-mind return that is opened or past the 14-day window.
    6. no_action — a delivery 10 days or less past its expected date (still in
       transit), or a digital/downloadable item that was delivered (digital
       goods are non-refundable unless never delivered).
    """

    case: str = dspy.InputField(desc="the customer's support case, including the facts needed")
    resolution: str = dspy.OutputField(
        desc="one of: refund, replacement, store_credit, return_label, escalate, no_action"
    )


class PolicyRouter(dspy.Module):
    """Resolves a support case to an action by following company policy.

    Args:
        instructions: Optional override for the resolver's initial prompt (used
            by the weak-seed demo mode so GEPA starts without the policy rules).
    """

    def __init__(self, instructions: str | None = None) -> None:
        super().__init__()
        signature = ResolveSupportCase
        if instructions:
            signature = signature.with_instructions(instructions)
        self.resolve = dspy.Predict(signature)

    def forward(self, case: str) -> dspy.Prediction:
        result = self.resolve(case=case)
        resolution = (result.resolution or "").strip().lower().replace(" ", "_")
        if resolution not in VALID_RESOLUTIONS:
            resolution = "no_action"
        return dspy.Prediction(resolution=resolution)
