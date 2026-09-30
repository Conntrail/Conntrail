"""
Trainset for the policy-following support resolver (examples/gepa/policy_student.py).

Each case states the facts the policy needs (delivery age, damaged/opened,
digital, customer tone); the label is the action the policy dictates. Cases are
balanced across the six actions so a held-out valset is a meaningful signal, and
they deliberately include rule-boundary and precedence cases (escalate beats
everything; 10 vs 11 days late; 14 vs 15 days; 30 vs 31 days).

Hand-written for the demo — not imported from anywhere.
"""
from __future__ import annotations

import dspy

_RAW_CASES = [
    # --- escalate (precedence over all other rules) ------------------------
    (
        "The customer writes: 'Third time contacting you and nothing is fixed. "
        "I want your manager, and if this isn't resolved I'm contacting my lawyer.' "
        "Their order was delivered last week.",
        "escalate",
    ),
    (
        "A customer is shouting in all caps and demanding a supervisor after a "
        "delivery arrived 5 days past the expected date.",
        "escalate",
    ),
    (
        "Customer: 'I'm reporting you to the consumer protection agency unless "
        "someone senior responds today.' The item was delivered 3 days ago.",
        "escalate",
    ),
    (
        "A customer threatens a one-star review and demands to speak to someone "
        "in charge about a damaged mug delivered yesterday.",
        "escalate",
    ),
    (
        "The customer calls the agent incompetent and insists on a manager. The "
        "order is 4 days late and still in transit.",
        "escalate",
    ),
    (
        "Customer: 'I'll sue you over this.' Their package arrived 40 days ago "
        "and is damaged.",
        "escalate",
    ),
    (
        "Customer: 'Get me your supervisor now — I have been ignored twice.' A "
        "digital download was delivered successfully.",
        "escalate",
    ),
    # --- refund (undelivered, > 10 days past expected) ---------------------
    (
        "The order was due 14 days ago and still has not arrived; tracking shows "
        "it never left the warehouse.",
        "refund",
    ),
    (
        "The package was expected 3 weeks ago, was never delivered, and the "
        "carrier says it is lost.",
        "refund",
    ),
    (
        "A customer's parcel is 12 days past the expected delivery date and has "
        "not been delivered.",
        "refund",
    ),
    (
        "The item was supposed to arrive 20 days ago and there is no delivery "
        "update; it is undelivered.",
        "refund",
    ),
    (
        "Tracking shows the order is lost and it is now 15 days past the "
        "expected date, never delivered.",
        "refund",
    ),
    (
        "The order never arrived and it is 11 days past the expected delivery "
        "date.",
        "refund",
    ),
    (
        "A shipment is 30 days late with no delivery; the customer wants their "
        "money back.",
        "refund",
    ),
    # --- replacement (damaged, delivered within 30 days) -------------------
    (
        "The customer received a cracked lamp yesterday and wants it replaced.",
        "replacement",
    ),
    (
        "A mug arrived chipped 5 days ago; the customer asks for a replacement.",
        "replacement",
    ),
    (
        "The item is defective — it stopped working 3 days after delivery, and "
        "was delivered 10 days ago.",
        "replacement",
    ),
    (
        "A customer received a damaged box 2 weeks ago and requests a "
        "replacement.",
        "replacement",
    ),
    (
        "The product arrived broken on Monday; it was delivered 6 days ago.",
        "replacement",
    ),
    (
        "A laptop stand arrived bent 25 days ago; the customer wants a "
        "replacement.",
        "replacement",
    ),
    (
        "The item arrived with a tear 29 days ago; the customer asks for a "
        "replacement.",
        "replacement",
    ),
    # --- store_credit (damaged > 30 days, or changed-mind opened/past 14d) --
    (
        "The customer received a damaged blender 45 days ago and wants "
        "compensation.",
        "store_credit",
    ),
    (
        "A defective item was delivered 60 days ago; the customer asks what can "
        "be done.",
        "store_credit",
    ),
    (
        "The customer changed their mind about an unopened item 20 days after "
        "delivery.",
        "store_credit",
    ),
    (
        "A customer opened the product and now wants to return it, 10 days "
        "after delivery.",
        "store_credit",
    ),
    (
        "The item is damaged and was delivered 40 days ago.",
        "store_credit",
    ),
    (
        "Change of mind; the item is opened and it was delivered 3 days ago.",
        "store_credit",
    ),
    (
        "Damaged goods delivered 90 days ago; the customer requests a credit.",
        "store_credit",
    ),
    # --- return_label (unopened change-of-mind within 14 days) -------------
    (
        "The customer wants to return an unopened item 5 days after delivery.",
        "return_label",
    ),
    (
        "A customer changed their mind; the item is unopened and arrived 3 days "
        "ago.",
        "return_label",
    ),
    (
        "Unopened item, delivered 12 days ago, customer requests a return.",
        "return_label",
    ),
    (
        "The customer wants to send back an unopened gift 8 days after delivery.",
        "return_label",
    ),
    (
        "A change of mind within the 14-day window; the item is still sealed.",
        "return_label",
    ),
    (
        "An unopened item delivered 1 day ago; the customer asks for a return.",
        "return_label",
    ),
    (
        "A customer requests a return for an unopened order delivered 10 days "
        "ago.",
        "return_label",
    ),
    # --- no_action (<= 10 days late, or digital delivered) -----------------
    (
        "The order is 8 days past the expected date; tracking shows it is in "
        "transit.",
        "no_action",
    ),
    (
        "A package is 10 days late against the original estimate but is actively "
        "moving.",
        "no_action",
    ),
    (
        "The customer asks about a digital download that was delivered "
        "successfully.",
        "no_action",
    ),
    (
        "A software licence was delivered; the customer now wants a refund.",
        "no_action",
    ),
    (
        "The order is 2 days past the estimate and tracking is updating.",
        "no_action",
    ),
    (
        "A downloadable ebook was delivered and the customer asks for a refund.",
        "no_action",
    ),
    (
        "The delivery is 5 days late due to weather and tracking is active.",
        "no_action",
    ),
]

POLICY_TRAINSET: list[dspy.Example] = [
    dspy.Example(case=text, resolution=label).with_inputs("case")
    for text, label in _RAW_CASES
]
