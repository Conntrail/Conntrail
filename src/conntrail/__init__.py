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

The public surface is intentionally small: the tracing entry points, the
config object, and the data types/tools you need to consume records and build
on top of the SDK (e.g. a CI gate or a report generator). Everything else is
an implementation detail.
"""
from conntrail.analyser import AnalysisResult, RetryExhaustedError
from conntrail.config import ConntrailConfig
from conntrail.contrast import ContrastGenerationError, ContrastSet
from conntrail.cost_analyzer import analyze_cost
from conntrail.exporters.base import BaseExporter
from conntrail.exporters.http import HttpExporter
from conntrail.exporters.recording import RecordingExporter
from conntrail.record import TraceRecord
from conntrail.utils.entropy import routing_entropy
from conntrail.utils.providers import DEFAULT_CONTRAST_MODEL, DEFAULT_MODEL, get_chat_model
from conntrail.wrap import trace_graph, trace_node

__all__ = [
    "trace_node",
    "trace_graph",
    "ConntrailConfig",
    "TraceRecord",
    "AnalysisResult",
    "ContrastSet",
    "ContrastGenerationError",
    "RetryExhaustedError",
    "BaseExporter",
    "HttpExporter",
    "RecordingExporter",
    "routing_entropy",
    "analyze_cost",
    "get_chat_model",
    "DEFAULT_MODEL",
    "DEFAULT_CONTRAST_MODEL",
]
__version__ = "0.1.0"
