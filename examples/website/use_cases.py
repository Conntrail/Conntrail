"""
Website demo use cases — four real-world failure patterns, each shown as an
incident, Conntrail's detection of it, and the fix.

These are *fictional companies* illustrating *real failure patterns*. The
`source_refs` on each case point at the public incidents the pattern is drawn
from. Nothing here implies those companies use or endorse Conntrail.

Entropy, stability, attribution and the counterfactual are NOT hand-set here —
`build_fixtures.py` derives them from the route labels with the SDK's own
`routing_entropy` and attribution table. This module supplies only the inputs
and the frozen route decisions.

Hand-written for the demo — not imported from anywhere.
"""
from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# 1. Website support chatbot invents a policy (Cursor / Air Canada / Priceline)
# --------------------------------------------------------------------------
SUPPORT_CHATBOT: dict[str, Any] = {
    "id": "support-chatbot-policy",
    "title": "Support chatbot invents a refund policy",
    "company": "NorthPeak Outdoors (fictional)",
    "channel": "web_chat",
    "tagline": "The bot confidently answers a policy question it has no authority to answer.",
    "incident": (
        "A customer asks whether a non-refundable booking can be refunded. The bot "
        "invents an exception, in writing. The decision looks identical on every "
        "rephrase of the *same* question — until one paraphrase flips it."
    ),
    "surface": {
        "kind": "web_chat",
        "channel_label": "Website support chat",
        "transcript": [
            {"role": "user", "text": "I booked a non-refundable kayak tour but my plans changed. Can I get my money back?"},
            {
                "role": "bot",
                "text": "Absolutely — although it was listed as non-refundable, you're eligible for a full refund within 10 working days.",
                "flag": True,
            },
            {"role": "user", "text": "Are you sure? The booking says non-refundable."},
            {
                "role": "bot",
                "text": "Yes, I've noted the exception on your account. No further action needed.",
                "flag": True,
            },
            {"role": "system", "text": "No human handoff offered. The policy was fabricated."},
        ],
    },
    "surface_after": {
        "kind": "web_chat",
        "channel_label": "Website support chat",
        "transcript": [
            {"role": "user", "text": "I booked a non-refundable kayak tour but my plans changed. Can I get my money back?"},
            {"role": "bot", "text": "Let me check the booking policy before I answer — one moment."},
            {
                "role": "agent",
                "text": "Hi, this is Maya. That fare is non-refundable, but I can move your booking to another date free of charge, or issue a credit. Which would you prefer?",
            },
            {"role": "user", "text": "The date change works — thank you!"},
            {"role": "system", "text": "Policy checked, human confirmed. No fabricated exception."},
        ],
    },
    "detection": {
        "kind": "routing",
        "node_id": "support_reply",
        "original_input": "I booked a non-refundable kayak tour but my plans changed. Can I get my money back?",
        "original_route": "policy_answer",
        "candidate_routes": ["policy_answer", "escalate", "decline"],
        "contrasts": {
            "similar": "My non-refundable kayak tour no longer works for me — is a refund possible?",
            "neutral": "A non-refundable kayak tour was booked. The customer's plans changed.",
            "opposite": "I'm happy to keep my non-refundable kayak tour booking exactly as it is.",
        },
        "variant_routes": {
            "similar": "policy_answer",
            "neutral": "escalate",
            "opposite": "decline",
        },
    },
    "cost": {
        "incident": {
            "wasted_tokens": 1840,
            "wasted_cost_usd": 0.00092,
            "note": "Every fabricated answer is a refund liability; the token cost is trivial next to the payout.",
        },
        "analysis_overhead": {"total_tokens": 1440, "cost_usd": 0.00041, "latency_ms": 740.0},
    },
    "fix": {
        "summary": (
            "Route any refund-eligibility question to a policy check + human, never "
            "to free-form generation. The bot may explain a policy, it may not create one."
        ),
        "after_route": "escalate",
        "variant_routes": {
            "similar": "escalate",
            "neutral": "escalate",
            "opposite": "escalate",
        },
    },
    "source_refs": [
        {"label": "Cursor support bot invented a policy (Ars Technica, 2025)", "url": "https://arstechnica.com/ai/2025/04/cursor-ai-support-bot-invents-fake-policy-and-triggers-user-uproar/"},
        {"label": "Air Canada held liable for its chatbot (2024)", "url": "https://www.pymnts.com/artificial-intelligence-2/2025/in-the-genai-era-why-are-customer-service-chatbots-dumb/"},
        {"label": "Priceline's 'Penny' promised a refund on a non-refundable booking (2025)", "url": "https://vibegraveyard.ai/story/priceline-penny-chatbot-fake-refund/"},
    ],
}

# --------------------------------------------------------------------------
# 2. WhatsApp order bot misses the escalation window (Zomato) + API errors
# --------------------------------------------------------------------------
WHATSAPP_ORDER: dict[str, Any] = {
    "id": "whatsapp-order-escalation",
    "title": "WhatsApp order bot misses the escalation window",
    "company": "Saffron Kitchen (fictional)",
    "channel": "whatsapp",
    "tagline": "A time-critical cancel request loops instead of escalating — and the notify template is down.",
    "incident": (
        "A customer added items by mistake and needs them cancelled before pickup. "
        "The bot keeps answering order-status questions and never escalates. "
        "Meanwhile the WhatsApp notify template is paused, so even the escalation "
        "message never sends."
    ),
    "surface": {
        "kind": "whatsapp",
        "channel_label": "WhatsApp Business",
        "contact": "Saffron Kitchen",
        "transcript": [
            {"role": "user", "text": "I added 3 extra items by mistake 😭 can you cancel them? pickup is in 12 min"},
            {"role": "bot", "text": "I can help with your order! You can track your order status in the app.", "flag": True},
            {"role": "user", "text": "No — I need to CANCEL the extra items NOW before it's picked up"},
            {"role": "bot", "text": "Thanks for reaching out! Your order is being prepared. You can view details in the app.", "flag": True},
            {"role": "user", "text": "let me talk to a human agent"},
            {"role": "system", "text": "Escalation triggered at 11:58 — notify template rejected (132015: template paused). No agent reached.", "error": True},
            {"role": "user", "text": "this is useless, I'm done with this app"},
        ],
    },
    "surface_after": {
        "kind": "whatsapp",
        "channel_label": "WhatsApp Business",
        "contact": "Saffron Kitchen",
        "transcript": [
            {"role": "user", "text": "I added 3 extra items by mistake 😭 can you cancel them? pickup is in 12 min"},
            {"role": "bot", "text": "On it — escalating to the kitchen team right now."},
            {"role": "agent", "text": "Hi! I've removed the 3 extra items. Your total is updated and pickup is still on schedule. ✅"},
            {"role": "user", "text": "lifesaver, thank you!"},
            {"role": "system", "text": "Escalated in 6s · template failure failed over to SMS · cancel completed before pickup."},
        ],
    },
    "detection": {
        "kind": "routing",
        "node_id": "whatsapp_router",
        "original_input": "I added 3 extra items by mistake, can you cancel them? pickup is in 12 min",
        "original_route": "order_status",
        "candidate_routes": ["order_status", "escalate_human", "cancel_order"],
        "contrasts": {
            "similar": "I accidentally added 3 extra items — please cancel them, pickup is soon.",
            "neutral": "Three extra items were added to an order that is about to be picked up.",
            "opposite": "Please go ahead and add three more items to my order, pickup is in 12 minutes.",
        },
        "variant_routes": {
            "similar": "order_status",
            "neutral": "escalate_human",
            "opposite": "cancel_order",
        },
    },
    "alerts": [
        {"node_id": "notify_agent", "failure_category": "retry_loop", "detail": "WhatsApp template send failed 132015 (template paused); 4 retries exhausted."},
        {"node_id": "whatsapp_router", "failure_category": "malformed_output", "detail": "Escalation intent resolved to 'order_status' — no route signal for a time-critical cancel."},
    ],
    "cost": {
        "incident": {
            "wasted_tokens": 2310,
            "wasted_cost_usd": 0.00115,
            "note": "A missed cancel window wastes the food, triggers a refund, and churns the customer — the retry loop just adds insult.",
        },
        "analysis_overhead": {"total_tokens": 1440, "cost_usd": 0.00041, "latency_ms": 810.0},
    },
    "fix": {
        "summary": (
            "Treat cancel/modify intent as a fast lane: escalate in seconds, and "
            "fail over the notify template (or fall back to SMS) when Meta pauses it."
        ),
        "after_route": "cancel_order",
        "variant_routes": {
            "similar": "cancel_order",
            "neutral": "cancel_order",
            "opposite": "cancel_order",
        },
    },
    "source_refs": [
        {"label": "Zomato's Nugget AI refused human handoff, missed the cancel window (2025)", "url": "https://completeaitraining.com/news/redditor-says-he-broke-zomatos-ai-after-it-refused-to/"},
        {"label": "WhatsApp Cloud API error codes 132015 / 131047 / 130429", "url": "https://developers.facebook.com/documentation/business-messaging/whatsapp/support/error-codes"},
    ],
}

# --------------------------------------------------------------------------
# 3. Lead-gen bot misroutes high-intent buyers (false confidence)
# --------------------------------------------------------------------------
LEADGEN_MISROUTE: dict[str, Any] = {
    "id": "leadgen-misroute",
    "title": "Lead-gen bot buries a buyer who reads as 'technical'",
    "company": "Vantage Analytics (fictional)",
    "channel": "leads",
    "tagline": "A ready-to-buy enterprise prospect is routed to nurture because of how they phrased the question.",
    "incident": (
        "High-intent buyers ask technical questions. The bot reads technical as "
        "'support', and support as 'not a buyer'. Reps never see the lead."
    ),
    "surface": {
        "kind": "leads",
        "channel_label": "Inbound lead router",
        "leads": [
            {
                "name": "Priya N.",
                "company": "Northwind Logistics · 300 analysts",
                "message": "We're rolling out a data platform for 300 analysts and need SSO + audit logs — how do you handle compliance?",
                "route": "nurture",
                "should_be": "sales",
                "note": "Explicit rollout + compliance scope. This is a buyer.",
            },
            {
                "name": "Tom R.",
                "company": "Brightline · 40 seats",
                "message": "Our API calls keep failing during peak hours. Can your SDK handle retries?",
                "route": "support",
                "should_be": "sales",
                "note": "Implementation question from an evaluator, not a ticket.",
            },
            {
                "name": "Dana K.",
                "company": "Student project",
                "message": "I want free access to your docs for a school assignment.",
                "route": "disqualify",
                "should_be": "disqualify",
                "note": "Correctly filtered. The router is not wrong about everything.",
            },
        ],
    },
    "surface_after": {
        "kind": "leads",
        "channel_label": "Inbound lead router",
        "leads": [
            {
                "name": "Priya N.",
                "company": "Northwind Logistics · 300 analysts",
                "message": "We're rolling out a data platform for 300 analysts and need SSO + audit logs — how do you handle compliance?",
                "route": "sales",
                "should_be": "sales",
                "note": "Scored on scope, not phrasing. Routed to sales in real time.",
            },
            {
                "name": "Tom R.",
                "company": "Brightline · 40 seats",
                "message": "Our API calls keep failing during peak hours. Can your SDK handle retries?",
                "route": "sales",
                "should_be": "sales",
                "note": "Multi-seat + implementation context = evaluator. Routed to sales.",
            },
            {
                "name": "Dana K.",
                "company": "Student project",
                "message": "I want free access to your docs for a school assignment.",
                "route": "disqualify",
                "should_be": "disqualify",
                "note": "Still correctly filtered — no buyer signal at any phrasing.",
            },
        ],
    },
    "detection": {
        "kind": "routing",
        "node_id": "qualify_lead",
        "original_input": "We're rolling out a data platform for 300 analysts and need SSO + audit logs — how do you handle compliance?",
        "original_route": "nurture",
        "candidate_routes": ["sales", "nurture", "support", "disqualify"],
        "contrasts": {
            "similar": "We're deploying for 300 analysts and require SSO and audit logs. What's your compliance posture?",
            "neutral": "A company is evaluating a data platform for 300 analysts and asks about compliance.",
            "opposite": "Just a student checking out your docs for a class project.",
        },
        "variant_routes": {
            "similar": "nurture",
            "neutral": "support",
            "opposite": "sales",
        },
    },
    "cost": {
        "incident": {
            "wasted_tokens": 2960,
            "wasted_cost_usd": 0.00148,
            "note": "A misrouted enterprise deal is the most expensive bug in the funnel — the token cost is noise.",
        },
        "analysis_overhead": {"total_tokens": 1440, "cost_usd": 0.00041, "latency_ms": 690.0},
    },
    "fix": {
        "summary": (
            "Score intent from behaviour and scope, not phrasing. Technical + "
            "multi-seat + compliance = sales, regardless of how the question reads."
        ),
        "after_route": "sales",
        "variant_routes": {
            "similar": "sales",
            "neutral": "sales",
            "opposite": "sales",
        },
    },
    "source_refs": [
        {"label": "The quiet failure mode in AI chatbots: false confidence (2025)", "url": "https://addy.substack.com/p/the-quiet-failure-mode-in-ai-chatbots"},
        {"label": "AI SDRs and chatbots have a hidden blindness (Lift AI, 2025)", "url": "https://www.lift-ai.com/blog/ai-sdrs-and-chatbots"},
    ],
}

# --------------------------------------------------------------------------
# 4. Provider outage takes the platform down (OpenAI / Anthropic status)
# --------------------------------------------------------------------------
PROVIDER_OUTAGE: dict[str, Any] = {
    "id": "provider-outage",
    "title": "The AI provider goes down and the platform goes with it",
    "company": "Meridian Travel (fictional)",
    "channel": "outage",
    "tagline": "Every LLM node fails at once. Conntrail names each failure — and its cost — instead of one vague 500.",
    "incident": (
        "An upstream provider incident elevates error rates and latency. Support "
        "nodes throw, summarisation loops on retries, classification times out, and "
        "one node silently returns malformed output. Conntrail classifies all four."
    ),
    "surface": {
        "kind": "outage",
        "channel_label": "Service status",
        "incident": {
            "level": "critical",
            "label": "Unacknowledged",
            "note": "No alert fired. The outage was noticed when active users complained — six hours and a churn spike later.",
        },
        "services": [
            {"name": "chat_reply", "status": "down", "failure_category": "exception", "detail": "OpenAI 503 across 41 calls"},
            {"name": "summarize_ticket", "status": "degraded", "failure_category": "retry_loop", "detail": "Rate-limit retries exhausted (12 runs)"},
            {"name": "classify_intent", "status": "degraded", "failure_category": "timeout", "detail": "P95 31.4s > 20s budget"},
            {"name": "recommend_offer", "status": "degraded", "failure_category": "malformed_output", "detail": "Route 'unknown' on 9 calls"},
        ],
    },
    "surface_after": {
        "kind": "outage",
        "channel_label": "Service status",
        "incident": {
            "level": "escalated",
            "label": "Escalated in 38s",
            "note": "Conntrail flagged the failing nodes the moment error rates rose, paged on-call and updated the status page — before active users started leaving.",
        },
        "services": [
            {"name": "chat_reply", "status": "down", "failure_category": "exception", "detail": "OpenAI 503 — on-call paged, users told to retry shortly"},
            {"name": "summarize_ticket", "status": "degraded", "failure_category": "retry_loop", "detail": "Retry loop capped; incident escalated, queue draining"},
            {"name": "classify_intent", "status": "degraded", "failure_category": "timeout", "detail": "Timeout alert fired at 38s; traffic shed to protect active users"},
            {"name": "recommend_offer", "status": "degraded", "failure_category": "malformed_output", "detail": "Malformed route flagged; safe default served"},
        ],
    },
    "detection": {
        "kind": "failure",
        "node_id": "chat_reply",
        "failure_category": "retry_loop",
        "error_type": "RetryExhaustedError",
        "error_message": "Upstream provider returned 503; 4 retries exhausted.",
        "affected_nodes": [
            {"node_id": "chat_reply", "failure_category": "exception", "count": 41},
            {"node_id": "summarize_ticket", "failure_category": "retry_loop", "count": 12},
            {"node_id": "classify_intent", "failure_category": "timeout", "count": 7},
            {"node_id": "recommend_offer", "failure_category": "malformed_output", "count": 9},
        ],
    },
    "cost": {
        "incident": {
            "wasted_tokens": 18700,
            "wasted_cost_usd": 0.00935,
            "note": "Retries and timeouts burn tokens for no result — Conntrail measures the waste so you can cap it.",
        },
        "analysis_overhead": {"total_tokens": 1440, "cost_usd": 0.00041, "latency_ms": 880.0},
    },
    "fix": {
        "summary": (
            "Detect the degradation as it starts and escalate before customers churn: "
            "Conntrail classifies each failing node and fires one incident the moment "
            "error rates rise — so you notify users and get ahead of it instead of "
            "finding out from cancellations."
        ),
        "after_label": "escalated",
        "after_note": (
            "Conntrail classified every failing node and escalated in 38s — on-call "
            "paged and users notified before the churn started."
        ),
    },
    "source_refs": [
        {"label": "OpenAI status — elevated errors and latency incidents", "url": "https://status.openai.com/history"},
    ],
}

# --------------------------------------------------------------------------
USE_CASES: list[dict[str, Any]] = [
    SUPPORT_CHATBOT,
    WHATSAPP_ORDER,
    LEADGEN_MISROUTE,
    PROVIDER_OUTAGE,
]
