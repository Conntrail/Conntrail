"""
Tests for trace_node() and trace_graph() (the public API).
"""
import pytest

from conntrail import ConntrailConfig, trace_graph, trace_node
from conntrail.interceptor import NodeInterceptor
from conntrail.wrap import _SKIP_NODES
from tests.fixtures.sample_graphs import build_simple_router


class TestTraceNode:
    def test_trace_node_import(self):
        assert callable(trace_node)

    def test_trace_graph_import(self):
        assert callable(trace_graph)

    def test_conntrail_config_import(self):
        assert ConntrailConfig is not None

    @pytest.mark.asyncio
    async def test_trace_node_does_not_alter_output(self):
        @trace_node(config=ConntrailConfig(sample_rate=0.0))
        async def my_node(state):
            return {**state, "route": "result"}

        out = await my_node({"message": "hello"})
        assert out["route"] == "result"

    @pytest.mark.asyncio
    async def test_trace_node_preserves_name(self):
        @trace_node(config=ConntrailConfig(sample_rate=0.0))
        async def classify_query(state):
            return state

        assert classify_query.__name__ == "classify_query"


class TestTraceGraph:
    def test_skip_nodes_excludes_start_and_end(self):
        assert _SKIP_NODES == {"__start__", "__end__"}

    def test_trace_graph_wraps_all_non_system_nodes(self):
        graph = build_simple_router()
        original_fns = {
            nid: node.bound.afunc or node.bound.func
            for nid, node in graph.nodes.items()
            if nid not in _SKIP_NODES
        }

        trace_graph(graph, config=ConntrailConfig(sample_rate=0.0))

        for nid, orig in original_fns.items():
            wrapped = graph.nodes[nid].bound.afunc
            assert isinstance(wrapped, NodeInterceptor), (
                f"Node {nid!r} was not wrapped by a NodeInterceptor"
            )
            assert wrapped.node_fn is orig

    def test_trace_graph_does_not_wrap_start_node(self):
        graph = build_simple_router()
        trace_graph(graph, config=ConntrailConfig(sample_rate=0.0))
        start_node = graph.nodes["__start__"].bound
        assert not isinstance(start_node.afunc, NodeInterceptor)
        assert not isinstance(start_node.func, NodeInterceptor)

    def test_trace_graph_respects_only_nodes(self):
        graph = build_simple_router()
        trace_graph(graph, config=ConntrailConfig(sample_rate=0.0), only_nodes={"router"})

        assert isinstance(graph.nodes["router"].bound.afunc, NodeInterceptor)
        assert not isinstance(graph.nodes["escalate_handler"].bound.afunc, NodeInterceptor)
        assert not isinstance(graph.nodes["general_handler"].bound.afunc, NodeInterceptor)

    @pytest.mark.asyncio
    async def test_trace_graph_does_not_alter_graph_output(self):
        """Graph output must be identical before and after trace_graph."""
        state = {"message": "I need this fixed ASAP", "route": None, "response": None}

        g1 = build_simple_router()
        result_before = await g1.ainvoke(state)

        g2 = build_simple_router()
        trace_graph(g2, config=ConntrailConfig(sample_rate=0.0))
        result_after = await g2.ainvoke(state)

        assert result_before["route"] == result_after["route"]
        assert result_before["response"] == result_after["response"]
