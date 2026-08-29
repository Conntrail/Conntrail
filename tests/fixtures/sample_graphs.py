"""
Minimal node/graph fixtures for unit testing.

Synthetic nodes and a synthetic compiled LangGraph with predictable routing
behaviour — used to test DivergenceAnalyser, NodeInterceptor, and the public
API without real LLM calls.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class RouterState(TypedDict):
    message: str
    route: str | None
    response: str | None


async def mock_router_node(state: RouterState) -> RouterState:
    """
    Deterministic router: "urgent" → escalate, everything else → general.
    Used to test entropy and attribution with predictable outcomes.
    """
    text = state["message"].lower()
    if any(word in text for word in ("urgent", "asap", "emergency", "critical", "immediately")):
        route = "escalate"
    else:
        route = "general"
    return {**state, "route": route}


def route_decision(state: RouterState) -> Literal["escalate_handler", "general_handler"]:
    return f"{state['route']}_handler"


async def escalate_handler(state: RouterState) -> RouterState:
    return {**state, "response": "Escalated."}


async def general_handler(state: RouterState) -> RouterState:
    return {**state, "response": "Handled generally."}


def build_simple_router() -> object:
    """Build a minimal 2-route LangGraph for testing."""
    builder = StateGraph(RouterState)
    builder.add_node("router", mock_router_node)
    builder.add_node("escalate_handler", escalate_handler)
    builder.add_node("general_handler", general_handler)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_decision,
        {"escalate_handler": "escalate_handler", "general_handler": "general_handler"},
    )
    builder.add_edge("escalate_handler", END)
    builder.add_edge("general_handler", END)

    return builder.compile()


# --- Routing test inputs ---
URGENT_INPUTS = [
    {"message": "I need this fixed ASAP, system is down", "route": None, "response": None},
    {"message": "This is an emergency! Help me immediately!", "route": None, "response": None},
    {"message": "CRITICAL issue — all services are failing", "route": None, "response": None},
]

ROUTINE_INPUTS = [
    {"message": "Can you help me with a general question?", "route": None, "response": None},
    {"message": "I would like some information about your services.", "route": None, "response": None},
    {"message": "What are your business hours?", "route": None, "response": None},
]
