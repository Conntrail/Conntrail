"""
Public API — trace_node decorator and trace_graph function.

Two integration patterns:

  Pattern 1: decorator on any node-shaped function (dict-in, dict-out) —
  works for a LangGraph node, but also for any other agent framework or a
  plain function you call by hand.
    @trace_node(config=ConntrailConfig())
    async def my_router_node(state): ...

  Pattern 2: wrap an entire compiled LangGraph graph in one call — the
  LangGraph-specific convenience adapter. For anything that isn't a
  compiled LangGraph graph, use trace_node()/NodeInterceptor directly instead.
    graph = trace_graph(compiled_graph, config=ConntrailConfig(sample_rate=0.2))
"""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from conntrail.config import ConntrailConfig
from conntrail.interceptor import NodeInterceptor, _active_trace_list

# LangGraph internal nodes that should never be traced
_SKIP_NODES = {"__start__", "__end__"}


def trace_node(
    config: ConntrailConfig | None = None,
    *,
    input_key: str = "message",
    route_key: str | None = None,
) -> Callable:
    """
    Decorator that wraps a single node-shaped function with Conntrail tracing.

    Generic — works on a LangGraph node, but equally on any other
    dict-in/dict-out function you want to observe (another framework's node,
    or a plain function called directly).

    Args:
        config: ConntrailConfig. Defaults to ConntrailConfig() if not provided.
        input_key: State key containing the text input for contrast generation.
        route_key: State key containing the routing decision. Auto-detected if None.

    Usage:
        @trace_node()
        async def my_node(state): ...

        @trace_node(config=ConntrailConfig(sample_rate=0.5))
        async def my_node(state): ...
    """
    resolved_config = config or ConntrailConfig()

    def decorator(fn: Callable) -> Callable:
        node_id = getattr(fn, "__name__", "unknown_node")
        interceptor = NodeInterceptor(
            fn,
            node_id=node_id,
            config=resolved_config,
            input_key=input_key,
            route_key=route_key,
        )

        @functools.wraps(fn)
        async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            return await interceptor(state)

        return wrapper

    return decorator


def trace_graph(
    compiled_graph: Any,
    config: ConntrailConfig | None = None,
    *,
    input_key: str = "message",
    route_key: str | None = None,
    only_nodes: set[str] | None = None,
) -> Any:
    """
    Wrap all nodes in a compiled LangGraph with Conntrail tracing.

    This is the LangGraph-specific convenience adapter — it patches each
    node's underlying callable in-place on a LangGraph CompiledStateGraph.
    For anything that isn't a compiled LangGraph graph, wrap the individual
    functions with trace_node() (or use NodeInterceptor directly) instead.

    Does not recompile the graph. Returns the same compiled_graph object.

    Args:
        compiled_graph: A LangGraph CompiledStateGraph instance.
        config: ConntrailConfig. Defaults to ConntrailConfig() if not provided.
        input_key: State key containing the text input for contrast generation.
        route_key: State key containing the routing decision. Auto-detected if None.
        only_nodes: Optional set of node names to trace. If None, all nodes are traced.
            Pass the routing node names to avoid tracing handler/leaf nodes.

    Returns:
        The same compiled_graph with all nodes wrapped.
    """
    resolved_config = config or ConntrailConfig()

    for node_id, pregel_node in compiled_graph.nodes.items():
        if node_id in _SKIP_NODES:
            continue
        if only_nodes is not None and node_id not in only_nodes:
            continue

        rc = pregel_node.bound

        if rc.afunc is not None:
            rc.afunc = NodeInterceptor(
                rc.afunc,
                node_id=node_id,
                config=resolved_config,
                input_key=input_key,
                route_key=route_key,
            )
        elif rc.func is not None:
            # Sync node: promote to async via the interceptor
            rc.afunc = NodeInterceptor(
                rc.func,
                node_id=node_id,
                config=resolved_config,
                input_key=input_key,
                route_key=route_key,
            )
            rc.func = None

    # Wrap ainvoke/invoke to collect traces out-of-band and inject into the result.
    # LangGraph strips unknown keys from node state updates, so traces must be
    # collected via ContextVar and added to the final result after the run completes.
    if not resolved_config.async_mode:
        _patch_invoke_methods(compiled_graph)

    return compiled_graph


def _patch_invoke_methods(compiled_graph: Any) -> None:
    """Wrap ainvoke and invoke to set up a per-call trace collector.

    LangGraph strips unknown keys from node state updates (TypedDict enforcement),
    so traces can't be injected via node return values. Instead, interceptors deposit
    records into a ContextVar, and these wrappers read that list after the run and
    merge it into the final result dict.
    """
    TRACE_KEY = NodeInterceptor.TRACE_KEY
    _orig_ainvoke = compiled_graph.ainvoke
    _orig_invoke = compiled_graph.invoke

    async def traced_ainvoke(input_state: Any, *args: Any, **kwargs: Any) -> Any:
        traces: list = []
        token = _active_trace_list.set(traces)
        try:
            result = await _orig_ainvoke(input_state, *args, **kwargs)
        finally:
            _active_trace_list.reset(token)
        if traces:
            result = {**result, TRACE_KEY: traces}
        return result

    def traced_invoke(input_state: Any, *args: Any, **kwargs: Any) -> Any:
        traces: list = []
        token = _active_trace_list.set(traces)
        try:
            result = _orig_invoke(input_state, *args, **kwargs)
        finally:
            _active_trace_list.reset(token)
        if traces:
            result = {**result, TRACE_KEY: traces}
        return result

    compiled_graph.ainvoke = traced_ainvoke
    compiled_graph.invoke = traced_invoke
