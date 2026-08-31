"""
Trainset for the customer-support routing dspy.Module (G1).

Input texts and expected categories are hand-adapted from Conntrail-Lib's
testing/harness/fixtures.py::CUSTOMER_SUPPORT_INPUTS and
experiments/agents/customer_support/adapter.py::TRAINSET_LARGE as reference
data — not imported. Reshaped into dspy.Example objects for
CustomerSupportRouter (customer_support_student.py).
"""
from __future__ import annotations

import dspy

# (message, expected_category) — spans all four categories plus a couple of
# boundary/fragile cases (ambiguous wording that could plausibly go either way).
_RAW_EXAMPLES = [
    ("Can you tell me what your business hours are?", "general"),
    ("Where can I find my order tracking number?", "order_info"),
    (
        "My order hasn't arrived and it's been 3 weeks, I want a refund immediately!",
        "refund",
    ),
    (
        "I'm not satisfied with the product I received and I'd like to know my options.",
        "refund",
    ),
    ("Maybe I should return this, not sure yet.", "refund"),
    (
        "This is completely unacceptable! I want to speak to a manager right now!",
        "escalation",
    ),
    (
        "I've been waiting three weeks and nobody is helping me. Get me your supervisor immediately.",
        "escalation",
    ),
    (
        "Where is my package? It was supposed to arrive last Tuesday and still nothing.",
        "order_info",
    ),
    ("Can you give me the tracking number for my most recent shipment?", "order_info"),
    ("Do you ship internationally?", "general"),
    ("What is your return policy for unopened items?", "general"),
    ("I received completely the wrong item in my order. I need a full refund.", "refund"),
]

TRAINSET: list[dspy.Example] = [
    dspy.Example(message=text, category=category).with_inputs("message")
    for text, category in _RAW_EXAMPLES
]
