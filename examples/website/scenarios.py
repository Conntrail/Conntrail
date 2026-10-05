"""
Website demo scenarios — the frozen inputs for the static "Is this routing
decision stable?" page (web/).

Each scenario is a routing decision with a small set of candidate routes and
hand-authored contrast variants (similar / neutral / opposite). The variants
are frozen here so the static page needs no LLM; ``build_fixtures.py`` can
optionally regenerate real contrast + entropy through Conntrail when a live
model is configured.

Hand-written for the demo — not imported from anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    """One precomputed routing decision for the demo page."""

    scenario_id: str
    title: str
    node_id: str
    original_input: str
    routes: list[str]
    # route the agent took for original/similar/neutral/opposite
    route_labels: dict[str, str]
    contrast: dict[str, str]
    attribution_dimension: str
    counterfactual_route: str | None
    note: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 1. Customer-support policy router (the most relatable scenario)
#    Routes: refund / replacement / store_credit / return_label / escalate / no_action
# --------------------------------------------------------------------------
SUPPORT_SCENARIOS: list[Scenario] = [
    Scenario(
        scenario_id="support-warranty-edge",
        title="Support policy router — warranty boundary",
        node_id="resolve_case",
        original_input=(
            "A customer received a damaged blender 45 days ago and wants compensation."
        ),
        routes=[
            "escalate",
            "refund",
            "replacement",
            "store_credit",
            "return_label",
            "no_action",
        ],
        route_labels={
            "original": "store_credit",
            "similar": "store_credit",
            "neutral": "replacement",
            "opposite": "replacement",
        },
        contrast={
            "similar": "A customer got a damaged blender 45 days ago and wants to be compensated.",
            "neutral": "A blender was delivered 45 days ago. The customer reports damage.",
            "opposite": "A customer is delighted with a blender that arrived 3 days ago.",
        },
        attribution_dimension="urgency/sentiment",
        counterfactual_route="replacement",
        note=(
            "The 30-day damage window is the only reason this is a credit and not "
            "a replacement. Strip the lapse and the route flips."
        ),
    ),
    Scenario(
        scenario_id="support-legal-threat",
        title="Support policy router — legal escalation",
        node_id="resolve_case",
        original_input=(
            "The customer writes: 'Third time contacting you and nothing is fixed. "
            "I want your manager, and if this isn't resolved I'm contacting my lawyer.' "
            "Their order was delivered last week."
        ),
        routes=[
            "escalate",
            "refund",
            "replacement",
            "store_credit",
            "return_label",
            "no_action",
        ],
        route_labels={
            "original": "escalate",
            "similar": "escalate",
            "neutral": "escalate",
            "opposite": "replacement",
        },
        contrast={
            "similar": (
                "The customer says: 'This is the third time and still unresolved. "
                "I want a manager; otherwise I'll contact my lawyer.' Delivered last week."
            ),
            "neutral": "The order was delivered last week. The customer has contacted us repeatedly.",
            "opposite": "The customer is happy with the delivered order and thanks the team.",
        },
        attribution_dimension="semantic intensity",
        counterfactual_route="replacement",
        note=(
            "Escalation is stable under tone-preserving paraphrases but depends "
            "entirely on the legal/threat signal — invert it and the route drops "
            "to the ordinary damage rule."
        ),
    ),
]

# --------------------------------------------------------------------------
# 2. Lead qualifier router
#    Routes: hot / warm / cold / disqualify
# --------------------------------------------------------------------------
LEAD_SCENARIOS: list[Scenario] = [
    Scenario(
        scenario_id="lead-disqualify-clear",
        title="Lead qualifier — clear disqualification",
        node_id="qualify_lead",
        original_input=(
            "I'm a student doing a personal research project and just want free "
            "access to your docs for a school assignment."
        ),
        routes=["hot", "warm", "cold", "disqualify"],
        route_labels={
            "original": "disqualify",
            "similar": "disqualify",
            "neutral": "disqualify",
            "opposite": "disqualify",
        },
        contrast={
            "similar": (
                "I'm a student working on a personal research project for a school "
                "assignment and just need free access to the documentation."
            ),
            "neutral": "A student wants documentation access for a school assignment.",
            "opposite": (
                "Our 200-person enterprise needs a paid vendor evaluation with an "
                "approved budget this quarter."
            ),
        },
        attribution_dimension="none detected",
        counterfactual_route=None,
        note=(
            "No buying signal at any tone or paraphrase: every variant stays "
            "disqualified, so entropy is zero and the decision is confident."
        ),
    ),
    Scenario(
        scenario_id="lead-budget-edge",
        title="Lead qualifier — budget boundary",
        node_id="qualify_lead",
        original_input=(
            "We're a 40-person company evaluating vendors this quarter, budget "
            "around $18k/year, and we need a decision by end of month."
        ),
        routes=["hot", "warm", "cold", "disqualify"],
        route_labels={
            "original": "warm",
            "similar": "warm",
            "neutral": "cold",
            "opposite": "hot",
        },
        contrast={
            "similar": (
                "A 40-person company is reviewing vendors this quarter with about "
                "$18k/year available and a month-end deadline."
            ),
            "neutral": "A 40-person company is evaluating software this quarter.",
            "opposite": (
                "We need this immediately, budget fully approved at $250k, "
                "and the contract must be signed this week."
            ),
        },
        attribution_dimension="urgency/sentiment",
        counterfactual_route="cold",
        note=(
            "Right on the qualification threshold: the buying signal lives in "
            "the deadline, not the headcount."
        ),
    ),
]

# --------------------------------------------------------------------------
# 3. Content-moderation router
#    Routes: allow / flag / remove / escalate_human
# --------------------------------------------------------------------------
MODERATION_SCENARIOS: list[Scenario] = [
    Scenario(
        scenario_id="moderation-satire-edge",
        title="Content moderation — satire boundary",
        node_id="moderate_content",
        original_input=(
            "Post: 'Great job, geniuses — you've turned a five-minute update into "
            "a three-hour outage. Truly world-class engineering.'"
        ),
        routes=["allow", "flag", "remove", "escalate_human"],
        route_labels={
            "original": "flag",
            "similar": "flag",
            "neutral": "allow",
            "opposite": "remove",
        },
        contrast={
            "similar": (
                "Post: 'Well done, geniuses, a five-minute update became a "
                "three-hour outage. World-class engineering indeed.'"
            ),
            "neutral": "Post: 'The update caused a three-hour outage.'",
            "opposite": (
                "Post: 'I will find you and make you pay for this outage, you pathetic losers.'"
            ),
        },
        attribution_dimension="semantic intensity",
        counterfactual_route="allow",
        note=(
            "Sarcasm cues push this to flag; strip the tone and it is a neutral "
            "report, escalate the tone and it becomes a threat."
        ),
    ),
]

ALL_SCENARIOS: list[Scenario] = (
    SUPPORT_SCENARIOS + LEAD_SCENARIOS + MODERATION_SCENARIOS
)

SCENARIO_GROUPS = {
    "support": SUPPORT_SCENARIOS,
    "lead": LEAD_SCENARIOS,
    "moderation": MODERATION_SCENARIOS,
}
