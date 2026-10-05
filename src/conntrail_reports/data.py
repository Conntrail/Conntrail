"""
AuditData — assemble the report's view model from collector data or a fixture
bundle.

Everything here is *derived* from the aggregates the collector already
produces (``/v1/cost-summary``) plus the per-trace records; nothing is
invented. The report recommends, it never claims a saving.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

_DIMENSION_ORDER = (
    "cache_efficiency",
    "prompt_size",
    "repeated_instructions",
    "output_discipline",
    "observer_overhead",
)

_DIMENSION_QUESTIONS = {
    "cache_efficiency": "Is the right stuff being cached?",
    "prompt_size": "Are prompts unnecessarily large?",
    "repeated_instructions": "Are identical instruction blocks re-sent without being cache-served?",
    "output_discipline": "Are output tokens held to a standard for label-like routes?",
    "observer_overhead": "What did Conntrail's own analysis cost?",
}

DEFAULT_PRICE_DISCLAIMER = (
    "Cost figures are estimates derived from the SDK's built-in price table "
    "(longest-prefix model match; local/* is free). Model prices drift — treat "
    "every figure as a signal, not an invoice. Corrections can be supplied via "
    "ConntrailConfig.model_prices or the CONNTRAIL_PRICE_OVERRIDES env var."
)


@dataclass
class FindingRollup:
    dimension: str
    question: str
    warning_count: int = 0
    info_count: int = 0
    examples: list[dict[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.warning_count + self.info_count


@dataclass
class AuditData:
    """The report's fully-derived view model."""

    generated_at: str
    source: str
    scanned_traces: int
    nodes: list[dict[str, Any]]
    shared_prompt_blocks: list[dict[str, Any]]
    findings: list[FindingRollup]
    totals: dict[str, Any]
    price_disclaimer: str = DEFAULT_PRICE_DISCLAIMER

    @property
    def deduped_node_ids(self) -> list[str]:
        return [n["node_id"] for n in self.nodes]


def _num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _aggregate_totals(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = sum(_num(n.get("input_tokens")) for n in nodes)
    cached = sum(_num(n.get("cached_input_tokens")) for n in nodes)
    output = sum(_num(n.get("output_tokens")) for n in nodes)
    cost = sum(_num(n.get("total_cost_usd")) for n in nodes)
    return {
        "input_tokens": int(input_tokens),
        "cached_input_tokens": int(cached),
        "output_tokens": int(output),
        "total_cost_usd": round(cost, 6),
        "cache_hit_ratio": round(cached / input_tokens, 3) if input_tokens else None,
        "warning_count": sum(int(_num(n.get("cost_warning_count"))) for n in nodes),
    }


def _rollup_findings(traces: list[dict[str, Any]]) -> list[FindingRollup]:
    by_dimension: dict[str, FindingRollup] = {
        dim: FindingRollup(dimension=dim, question=_DIMENSION_QUESTIONS.get(dim, ""))
        for dim in _DIMENSION_ORDER
    }
    for trace in traces:
        findings = trace.get("cost_findings") or []
        if not isinstance(findings, list):
            continue
        for finding in findings:
            dimension = finding.get("dimension")
            if dimension not in by_dimension:
                continue
            rollup = by_dimension[dimension]
            if finding.get("severity") == "warning":
                rollup.warning_count += 1
            else:
                rollup.info_count += 1
            if len(rollup.examples) < 3:
                rollup.examples.append(
                    {
                        "node_id": str(trace.get("node_id", "")),
                        "evidence": str(finding.get("evidence", "")),
                        "recommendation": str(finding.get("recommendation", "")),
                    }
                )
    return [by_dimension[dim] for dim in _DIMENSION_ORDER]


def build_audit_data(
    *,
    cost_summary: dict[str, Any],
    traces: list[dict[str, Any]],
    source: str = "collector",
    generated_at: str | None = None,
) -> AuditData:
    """Derive the report view model from a cost summary + trace records."""
    nodes = list(cost_summary.get("nodes") or [])
    shared = list(cost_summary.get("shared_prompt_blocks") or [])
    scanned = cost_summary.get("scanned_traces")
    if not isinstance(scanned, int):
        scanned = sum(int(_num(n.get("trace_count"))) for n in nodes)

    return AuditData(
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        source=source,
        scanned_traces=scanned,
        nodes=nodes,
        shared_prompt_blocks=shared,
        findings=_rollup_findings(traces),
        totals=_aggregate_totals(nodes),
    )


def audit_data_from_bundle(bundle: dict[str, Any]) -> AuditData:
    """Derive the report view model from a JSON fixture/export bundle.

    Expected bundle shape (produced by ``GET /v1/export`` or a fixture bundle
    with an added ``cost_summary``)::

        {"cost_summary": {...}, "traces": [ <TraceRecord.to_dict()>, ...]}
    """
    cost_summary = bundle.get("cost_summary")
    if not isinstance(cost_summary, dict):
        raise ValueError("bundle is missing a 'cost_summary' object")
    traces = bundle.get("traces") or []
    if not isinstance(traces, list):
        raise ValueError("bundle 'traces' must be a list")
    return build_audit_data(
        cost_summary=cost_summary,
        traces=traces,
        source="bundle",
        generated_at=bundle.get("generated_at"),
    )
