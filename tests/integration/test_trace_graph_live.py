"""
Integration test: trace_graph() against a real LLM (needs a live API key).

Fixed from the old repo's mismarking — this genuinely calls a live LLM
(via ContrastGenerator's contrast-generation pipeline) and is marked
@pytest.mark.integration from the start.
"""
import os

import pytest

from conntrail import ConntrailConfig, trace_graph
from tests.fixtures.sample_graphs import build_simple_router


@pytest.mark.integration
class TestTraceGraphLive:
    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not any(os.getenv(k) for k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
            pytest.skip("No API key available")

    @pytest.mark.asyncio
    async def test_trace_graph_does_not_alter_graph_output(self):
        """Graph output must be identical before and after trace_graph, even
        with a real contrast-analysis pipeline running behind it."""
        state = {"message": "I need this fixed ASAP, system is down", "route": None, "response": None}

        g1 = build_simple_router()
        result_before = await g1.ainvoke(state)

        g2 = build_simple_router()
        trace_graph(g2, config=ConntrailConfig(sample_rate=1.0, async_mode=True))
        result_after = await g2.ainvoke(state)

        assert result_before["route"] == result_after["route"]
        assert result_before["response"] == result_after["response"]
