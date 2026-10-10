"""
The routing decision the live demo analyses.

A small customer-support intent router: the visitor's message is classified
into exactly one route, and Conntrail measures how stable that classification
is under perturbation.
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

ROUTES = ("refund", "order_status", "escalate", "general")

NODE_ID = "support_router"

SYSTEM_PROMPT = """You are the routing layer of a customer-support agent.

Read the customer message and choose exactly ONE route:

- refund: the customer wants money back for an order.
- order_status: the customer asks where an order is or about delivery status.
- escalate: the customer is angry, threatens, or demands a human/manager.
- general: anything else.

Reply with ONLY the route word, in lowercase, and nothing else."""


def parse_route(text: str) -> str:
    """Extract the route from the model's reply (falls back to 'general')."""
    if not text:
        return "general"
    cleaned = re.sub(r"[^a-z_ ]", " ", text.lower())
    for token in cleaned.split():
        if token in ROUTES:
            return token
    return "general"


def make_router_node(llm: Any, *, input_key: str = "message", route_key: str = "route"):
    """Build the node_fn the analyser re-runs on each variant."""

    async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
        message = str(state.get(input_key, ""))
        response = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)]
        )
        route = parse_route(getattr(response, "content", "") or "")
        return {**state, route_key: route}

    return node_fn
