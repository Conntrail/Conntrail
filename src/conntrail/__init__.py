"""
Conntrail — observability for agentic workflows.

Traces node calls (LangGraph or otherwise), measures routing stability via
contrastive analysis, and exports records to a collector for storage and
review.

Quick start:
    from conntrail import trace_node, trace_graph, ConntrailConfig

    # Wrap a single node-shaped function (LangGraph node or otherwise):
    @trace_node(config=ConntrailConfig())
    async def my_router_node(state): ...

    # Wrap an entire compiled LangGraph graph (LangGraph-specific convenience adapter):
    graph = trace_graph(compiled_graph, config=ConntrailConfig(sample_rate=0.2))
"""
from conntrail.config import ConntrailConfig
from conntrail.wrap import trace_graph, trace_node

__all__ = ["trace_node", "trace_graph", "ConntrailConfig"]
__version__ = "0.1.0"
