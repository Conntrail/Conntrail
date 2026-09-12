"""
DivergenceAnalyser — runs original + 3 contrasts through a node and measures routing divergence.

Computes Shannon entropy over the 4 routing outcomes and looks up which semantic
dimension is associated with the decision (attribution).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from conntrail.contrast import ContrastSet
from conntrail.utils.entropy import routing_entropy

if TYPE_CHECKING:
    from conntrail.cost import TokenUsage

logger = logging.getLogger("conntrail")

# Fixed attribution labels by contrast dimension.
# The dimension that first flips the route names the attribution.
# This is a static priority lookup, not semantic inference — see
# DivergenceAnalyser._infer_attribution.
_ATTRIBUTION_LABELS = {
    "opposite": "semantic intensity",   # core dimension fully inverted → strongest signal
    "neutral":  "urgency/sentiment",    # stripping emphasis changed the route
    "similar":  "surface form",         # even a paraphrase flipped the route → very fragile
}


class RetryExhaustedError(Exception):
    """Raised when _call_node's rate-limit retry loop exhausts max_retries.

    Carries the original (rate-limit) exception so callers can inspect it,
    and gives NodeInterceptor/the failure classifier a distinguishable type
    to key off (classified as failure_category "retry_loop") instead of a
    bare, generically-typed Exception.
    """

    def __init__(self, original_error: Exception) -> None:
        self.original_error = original_error
        super().__init__(f"Retry loop exhausted after rate limiting: {original_error}")


@dataclass
class AnalysisResult:
    """
    Output of DivergenceAnalyser.analyse() for one node call.
    Consumed by NodeInterceptor to build a TraceRecord.
    """

    original_route: str
    contrast_routes: dict[str, str]   # {"similar": route, "neutral": route, "opposite": route}
    entropy_score: float
    attribution_dimension: str
    counterfactual_route: str | None
    raw_outputs: dict[str, Any]       # all 4 node outputs keyed by variant name
    # --- observer self-cost (the 4 re-runs' own LLM usage; None when the
    # --- node makes no observable LangChain LLM calls) ---
    analysis_usage: TokenUsage | None = None
    analysis_models: list[str] = field(default_factory=list)
    analysis_latency_ms: float | None = None
    retries: int = 0


@dataclass
class _CallStats:
    """Per-variant cost/latency stats filled in by _call_node."""

    latency_ms: float = 0.0
    retries: int = 0
    usage: TokenUsage | None = None
    models: list[str] = field(default_factory=list)


class DivergenceAnalyser:
    """
    Runs the traced node with original + 3 contrast inputs concurrently.
    Measures routing divergence and computes attribution.

    Routing comparison strategy: auto-detect the state key that changed
    between input and output and use its value as the route (see
    ``_extract_route``). Pass ``route_key`` explicitly to skip auto-detection.
    """

    async def analyse(
        self,
        node_fn: Callable,
        original_input: dict[str, Any],
        contrast_set: ContrastSet,
        input_key: str = "message",
        route_key: str | None = None,
    ) -> AnalysisResult:
        """
        Run all 4 inputs through node_fn concurrently and analyse divergence.

        Args:
            node_fn:        The LangGraph node function being traced.
            original_input: The original state dict passed to the node.
            contrast_set:   The 3 contrast variants to run.
            input_key:      Which state key holds the text input (default: "message").
            route_key:      Which output key holds the routing decision.
                            Auto-detected if None.

        Returns:
            AnalysisResult with entropy_score, attribution_dimension, and
            counterfactual_route.
        """
        variants: dict[str, dict[str, Any]] = {
            "original": original_input,
            "similar":  {**original_input, input_key: contrast_set.similar},
            "neutral":  {**original_input, input_key: contrast_set.neutral},
            "opposite": {**original_input, input_key: contrast_set.opposite},
        }

        # Run all 4 concurrently, capturing per-variant cost/latency stats.
        stats_by_name = {name: _CallStats() for name in variants}
        start = time.perf_counter()
        outputs_list = await asyncio.gather(
            *[
                self._call_node(node_fn, state, stats=stats_by_name[name])
                for name, state in variants.items()
            ]
        )
        analysis_latency_ms = (time.perf_counter() - start) * 1000
        raw_outputs: dict[str, Any] = dict(zip(variants.keys(), outputs_list))

        # Extract route label from each output
        routes: dict[str, str] = {
            name: self._extract_route(variants[name], output, route_key=route_key)
            for name, output in raw_outputs.items()
        }

        entropy = routing_entropy(list(routes.values()))
        contrast_routes = {k: v for k, v in routes.items() if k != "original"}
        attribution, counterfactual = self._infer_attribution(
            original_route=routes["original"],
            contrast_routes=contrast_routes,
        )

        from conntrail.cost import TokenUsage

        analysis_usage = TokenUsage.merged(
            s.usage for s in stats_by_name.values() if s.usage is not None
        )
        analysis_models = sorted({m for s in stats_by_name.values() for m in s.models})

        return AnalysisResult(
            original_route=routes["original"],
            contrast_routes=contrast_routes,
            entropy_score=entropy,
            attribution_dimension=attribution,
            counterfactual_route=counterfactual,
            raw_outputs=raw_outputs,
            analysis_usage=analysis_usage,
            analysis_models=analysis_models,
            analysis_latency_ms=analysis_latency_ms,
            retries=sum(s.retries for s in stats_by_name.values()),
        )

    async def _call_node(
        self, node_fn: Callable, state: dict[str, Any], stats: _CallStats | None = None
    ) -> Any:
        """Call node_fn with automatic retry on rate-limit errors (429).

        Parses the provider's retry-after hint when present (e.g. Groq's
        "Please try again in 2m5.3s" message) so retries respect the actual
        window rather than blind exponential backoff.

        When ``stats`` is provided, records wall-clock latency (including
        retry backoff — retries are latency too), retry count, and the
        node-internal LLM usage observed via a cost callback handler.

        Raises:
            RetryExhaustedError: if a rate-limit error persists through all
                max_retries attempts. Any other exception propagates as-is
                (it never entered a retry loop).
        """
        from conntrail.cost import CostCallbackHandler, llm_cost_capture

        max_retries = 4
        handler = CostCallbackHandler()
        start = time.perf_counter()
        try:
            with llm_cost_capture(handler):
                for attempt in range(max_retries):
                    try:
                        if inspect.iscoroutinefunction(node_fn):
                            return await node_fn(state)
                        return await asyncio.to_thread(node_fn, state)
                    except Exception as exc:
                        msg = str(exc)
                        is_rate_limit = (
                            "429" in msg
                            or "rate_limit" in msg.lower()
                            or "rate limit" in msg.lower()
                        )
                        if is_rate_limit and attempt < max_retries - 1:
                            if stats is not None:
                                stats.retries += 1
                            delay = self._parse_retry_after(msg) or (
                                5.0 * (2 ** attempt) + random.uniform(0, 1)
                            )
                            logger.debug(
                                "conntrail: rate limit on node call (attempt %d), retrying in %.1fs",
                                attempt + 1,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        if is_rate_limit:
                            if stats is not None:
                                stats.retries += 1
                            raise RetryExhaustedError(exc) from exc
                        raise
        finally:
            if stats is not None:
                stats.latency_ms = (time.perf_counter() - start) * 1000
                stats.usage = handler.total_usage()
                stats.models = sorted({c.model for c in handler.calls if c.model})

    @staticmethod
    def _parse_retry_after(error_msg: str) -> float | None:
        """Parse 'try again in Xm Y.Zs' or 'try again in Y.Zs' from an error message."""
        # e.g. "Please try again in 1m50.592s"
        m = re.search(r"try again in (?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", error_msg, re.IGNORECASE)
        if m:
            minutes = float(m.group(1) or 0)
            seconds = float(m.group(2))
            return minutes * 60 + seconds + 2.0  # add 2s buffer
        return None

    def _extract_route(
        self,
        input_state: dict[str, Any],
        output_state: dict[str, Any],
        route_key: str | None = None,
    ) -> str:
        """
        Extract the routing label from a node's output state.

        Strategy (in order):
          1. Use ``route_key`` directly if provided.
          2. Find a key that was None in input and is now a non-empty string.
          3. Find any string key whose value changed from input to output.
          4. Fall back to "unknown".
        """
        if route_key:
            value = output_state.get(route_key)
            return str(value) if value is not None else "unknown"

        # Strategy 2: None → string
        for key, value in output_state.items():
            if isinstance(value, str) and value and input_state.get(key) is None:
                return value

        # Strategy 3: any string key that changed
        for key, value in output_state.items():
            if isinstance(value, str) and value and value != input_state.get(key):
                return value

        return "unknown"

    def _infer_attribution(
        self,
        original_route: str,
        contrast_routes: dict[str, str],
    ) -> tuple[str, str | None]:
        """
        Look up the attribution dimension and counterfactual route from a
        fixed priority table.

        This is NOT semantic inference — it's a static
        opposite > neutral > similar lookup: whichever of the three checks
        first finds a route that differs from the original names the
        "driver" dimension. There is no analysis of *why* that dimension
        changed the route, just which one changed it first by priority.

        Returns:
            (attribution_dimension, counterfactual_route)
        """
        for dim in ("opposite", "neutral", "similar"):
            if contrast_routes.get(dim) != original_route:
                return _ATTRIBUTION_LABELS[dim], contrast_routes[dim]

        return "none detected", None
