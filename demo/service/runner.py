"""
Run the real Conntrail analysis for one message.

Uses the SDK's ContrastGenerator (1 call) and DivergenceAnalyser (4 node
re-runs) — exactly the path the library ships, not a reimplementation.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from conntrail.analyser import DivergenceAnalyser
from conntrail.contrast import ContrastGenerator
from conntrail.cost import build_analysis_overhead
from conntrail.record import TraceRecord
from demo.service.models import ModelConfig, build_chat_model
from demo.service.router import NODE_ID, ROUTES, make_router_node


async def run_analysis(message: str, configs: list[ModelConfig]) -> dict[str, Any]:
    """Analyse one message, trying each config in order.

    Free models are rate-limited or occasionally return unusable output, so we
    fall back through the candidate list before giving up.
    """
    last_error: Exception | None = None
    for cfg in configs:
        try:
            return await _run_once(message, cfg)
        except Exception as exc:  # noqa: BLE001 - deliberate fallback across models
            last_error = exc
    assert last_error is not None
    raise last_error


async def _run_once(message: str, cfg: ModelConfig) -> dict[str, Any]:
    """Run the real Conntrail analysis for one message and model config."""
    llm = build_chat_model(cfg)
    generator = ContrastGenerator(llm=llm)
    contrasts = await generator.generate(message)

    analyser = DivergenceAnalyser()
    node_fn = make_router_node(llm)
    result = await analyser.analyse(
        node_fn,
        {"message": message},
        contrasts,
        input_key="message",
        route_key="route",
    )

    overhead = build_analysis_overhead(
        re_run_usage=result.analysis_usage,
        re_run_models=result.analysis_models,
        re_run_latency_ms=result.analysis_latency_ms,
        retries=result.retries,
        contrast_usage=generator.last_usage,
        contrast_model=generator.last_model or cfg.model,
    )
    # The demo runs on free models only — the analysis genuinely costs $0, so
    # don't surface a price-table estimate that implies otherwise.
    if cfg.model.endswith(":free") or cfg.model == "openrouter/free":
        overhead["cost_usd"] = 0.0

    # Normalise -0.0 so the UI never renders "-0.00".
    entropy = round(result.entropy_score, 3) + 0.0
    stability = TraceRecord.stability_label(entropy)
    record = TraceRecord(
        trace_id=TraceRecord.make_id(),
        node_id=NODE_ID,
        timestamp=datetime.now(UTC),
        original_input=message,
        original_route=result.original_route,
        entropy_score=entropy,
        stability=stability,
        attribution_dimension=result.attribution_dimension,
        plain_language_summary=TraceRecord.build_summary(
            NODE_ID,
            result.original_route,
            stability,
            entropy,
            result.attribution_dimension,
            result.counterfactual_route,
        ),
        raw_contrasts=contrasts,
        raw_outputs=result.contrast_routes,
        counterfactual_route=result.counterfactual_route,
        analysis_overhead=overhead,
    )

    payload = record.to_dict()
    payload["candidate_routes"] = list(ROUTES)
    payload["model"] = cfg.model
    payload["provider"] = cfg.provider
    return payload
