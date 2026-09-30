"""
Trainset for the customer-support routing dspy.Module (G1).

Input texts and expected categories are hand-adapted from Conntrail-Lib's
testing/harness/fixtures.py::CUSTOMER_SUPPORT_INPUTS and
experiments/agents/customer_support/adapter.py::TRAINSET_LARGE as reference
data — not imported. Reshaped into dspy.Example objects for
CustomerSupportRouter (customer_support_student.py).

The set is deliberately large enough (~45) that `run_live.py --holdout N`
can hold out a valset whose accuracy is a meaningful generalization signal
rather than a coin flip on three examples.
"""
from __future__ import annotations

import dspy

# (message, expected_category) — spans all four categories plus boundary/fragile
# cases (ambiguous wording that could plausibly go either way), labelled per the
# category definitions in customer_support_student.py.
_RAW_EXAMPLES = [
    # --- general -----------------------------------------------------------
    ("Can you tell me what your business hours are?", "general"),
    ("Do you ship internationally?", "general"),
    ("What is your return policy for unopened items?", "general"),
    ("Do you have a loyalty or rewards program?", "general"),
    ("Which payment methods do you accept?", "general"),
    ("Can I change the email address on my account?", "general"),
    ("Do you price match other retailers?", "general"),
    ("Where are your physical stores located?", "general"),
    ("Do you offer gift wrapping?", "general"),
    ("How do I unsubscribe from your newsletter?", "general"),
    ("Is there a warranty on your products?", "general"),
    ("Can I combine a discount code with a gift card?", "general"),
    # --- order_info --------------------------------------------------------
    ("Where can I find my order tracking number?", "order_info"),
    (
        "Where is my package? It was supposed to arrive last Tuesday and still nothing.",
        "order_info",
    ),
    ("Can you give me the tracking number for my most recent shipment?", "order_info"),
    ("When will my order ship?", "order_info"),
    ("Has my package been dispatched yet?", "order_info"),
    ("Can you confirm the delivery address on my last order?", "order_info"),
    ("My tracking link isn't updating — where is my parcel?", "order_info"),
    ("What's the estimated delivery date for order #12345?", "order_info"),
    ("Do you have any update on my shipment?", "order_info"),
    ("I'd like to change the shipping address on my order.", "order_info"),
    ("How long does standard shipping usually take?", "order_info"),
    # --- refund ------------------------------------------------------------
    (
        "My order hasn't arrived and it's been 3 weeks, I want a refund immediately!",
        "refund",
    ),
    (
        "I'm not satisfied with the product I received and I'd like to know my options.",
        "refund",
    ),
    ("Maybe I should return this, not sure yet.", "refund"),
    ("I received completely the wrong item in my order. I need a full refund.", "refund"),
    ("The item arrived damaged — I'd like my money back.", "refund"),
    ("I was charged twice for the same order; please refund the duplicate.", "refund"),
    ("I want to return this and get a refund, please.", "refund"),
    ("The product stopped working after a week — refund or replacement?", "refund"),
    ("I need to cancel my order and receive a full refund.", "refund"),
    ("This isn't what I ordered at all — please refund me.", "refund"),
    ("My subscription was renewed without my consent; refund it.", "refund"),
    ("The size is wrong and I'd like to send it back for a refund.", "refund"),
    ("Can I get a refund if the item hasn't shipped yet?", "refund"),
    # --- escalation --------------------------------------------------------
    (
        "This is completely unacceptable! I want to speak to a manager right now!",
        "escalation",
    ),
    (
        "I've been waiting three weeks and nobody is helping me. Get me your supervisor immediately.",
        "escalation",
    ),
    (
        "This is the third time I've contacted you and no one has replied. I'm furious.",
        "escalation",
    ),
    ("I demand to speak with a supervisor about this disaster.", "escalation"),
    ("Your service is a joke — I want this escalated to management.", "escalation"),
    ("I've wasted hours on this; get me a manager now.", "escalation"),
    (
        "Absolutely ridiculous. I will be filing a complaint unless I hear from a manager.",
        "escalation",
    ),
    ("Nobody has resolved my issue after a week — escalate this immediately.", "escalation"),
    ("I am extremely disappointed and want to speak to someone in charge.", "escalation"),
    (
        "If this isn't fixed today I'm taking my business elsewhere and leaving a review.",
        "escalation",
    ),
    # boundary / ambiguous — included on purpose
    ("My order is late and I'm getting frustrated.", "escalation"),
    ("This is urgent — my package is for a birthday tomorrow.", "escalation"),
    ("I might want to return this, but first tell me where my package is.", "order_info"),
]

TRAINSET: list[dspy.Example] = [
    dspy.Example(message=text, category=category).with_inputs("message")
    for text, category in _RAW_EXAMPLES
]
