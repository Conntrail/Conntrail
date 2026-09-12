"""
CostAnalyzer — derives cost findings from captured cost telemetry.

Same philosophy as the server-side failure classifier: findings are derived
from signals the SDK already produces, never invented. Every finding is a
dict with ``dimension``, ``severity`` ("info" | "warning"), ``evidence``,
and ``recommendation``.

Dimensions and the questions they answer:

  cache_efficiency      — is the right stuff being cached? (hit rate, wasted
                          cache writes, missing cache_control on Anthropic)
  prompt_size           — are prompts unnecessarily large?
  repeated_instructions — are the same instruction blocks re-sent across the
                          node's LLM calls without being cache-served?
  output_discipline     — are output tokens held to a standard (label-like
                          routes should cost ~zero output tokens)?
  observer_overhead     — what Conntrail's own analysis cost for this trace.

Findings recommend, never mutate: research shows naive prompt compression
can crater accuracy (LLMLingua-2 destroys tool schemas; one optimizer
benchmarked 32% -> 8% accuracy at 50% token reduction), so the human — or
the GEPA reflection LM — decides what to act on.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conntrail.record import TraceRecord

# --- thresholds (tuned from provider cache minimums + published research) ---
_CACHEABLE_MIN_TOKENS = 1024      # provider prompt-cache eligibility floor
_LARGE_CACHE_PROMPT_TOKENS = 2048
_LOW_CACHE_HIT_RATIO = 0.2
_GOOD_CACHE_HIT_RATIO = 0.8
_LARGE_PROMPT_TOKENS = 8000       # "very large prompt" flag
_STATIC_DOMINANT_SHARE = 0.8      # system prompt share of total prompt chars
_STATIC_LARGE_TOKENS = 4000
_VERBOSE_OUTPUT_TOKENS = 25       # tokens/call for a label-like route
_LABEL_MAX_CHARS = 32             # route strings this short look enum-like
_OVERHEAD_DOMINANCE = 3.0         # observer tokens vs node tokens

_DIMENSION_ORDER = (
    "cache_efficiency",
    "prompt_size",
    "repeated_instructions",
    "output_discipline",
    "observer_overhead",
)


def _finding(
    dimension: str, severity: str, evidence: str, recommendation: str
) -> dict[str, str]:
    return {
        "dimension": dimension,
        "severity": severity,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def analyze_cost(record: TraceRecord) -> list[dict[str, str]]:
    """Derive cost findings for one trace record (empty list when nothing fires)."""
    findings: list[dict[str, str]] = []
    usage = record.token_usage if isinstance(record.token_usage, dict) else None

    findings.extend(_check_cache_efficiency(usage))
    findings.extend(_check_prompt_size(usage))
    findings.extend(_check_repeated_instructions(usage))
    findings.extend(_check_output_discipline(record, usage))
    findings.extend(_check_observer_overhead(record, usage))

    findings.sort(key=lambda f: _DIMENSION_ORDER.index(f["dimension"]))
    return findings


# --------------------------------------------------------------------------
# cache_efficiency
# --------------------------------------------------------------------------


def _check_cache_efficiency(usage: dict[str, Any] | None) -> list[dict[str, str]]:
    if not usage:
        return []
    input_tokens = usage.get("input_tokens") or 0
    if input_tokens < _CACHEABLE_MIN_TOKENS:
        return []

    calls = usage.get("llm_call_count") or 0
    cached = usage.get("cached_input_tokens") or 0
    written = usage.get("cache_write_tokens") or 0
    models = usage.get("models") or []
    hit_ratio = cached / input_tokens if input_tokens else 0.0
    findings: list[dict[str, str]] = []

    # Anthropic requires explicit cache_control breakpoints — zero cache
    # activity on a cacheable prompt means they're almost certainly missing.
    if calls >= 2 and cached == 0 and written == 0 and any(
        m.lower().startswith("claude") for m in models
    ):
        findings.append(
            _finding(
                "cache_efficiency",
                "warning",
                f"{calls} LLM calls with ~{input_tokens} input tokens on "
                f"Anthropic show zero cache activity (0 reads, 0 writes).",
                "Anthropic caching is opt-in: add cache_control breakpoints on "
                "the stable prompt prefix (reads bill at ~0.1x the input rate).",
            )
        )

    # Cache writes that are never read: dynamic content is being written to
    # the cache and invalidated before reuse ("Don't Break the Cache",
    # arXiv:2601.06007 — naive full-context caching can regress latency).
    if written > 0 and cached == 0 and calls >= 2:
        findings.append(
            _finding(
                "cache_efficiency",
                "warning",
                f"{written} cache-write tokens with 0 cache reads across {calls} "
                "calls — the cached prefix appears to change between calls.",
                "Move dynamic content (timestamps, per-request data) to the end "
                "of the prompt and cache only the stable prefix; cache writes "
                "cost a premium (1.25x on Anthropic) and are wasted when never "
                "re-read.",
            )
        )

    # Low hit rate on large prompts.
    if (cached > 0 or written > 0) and calls >= 2:
        if input_tokens >= _LARGE_CACHE_PROMPT_TOKENS and hit_ratio < _LOW_CACHE_HIT_RATIO:
            findings.append(
                _finding(
                    "cache_efficiency",
                    "warning",
                    f"Cache hit rate is {hit_ratio:.0%} on ~{input_tokens}-token "
                    f"prompts ({cached} of {input_tokens} input tokens served "
                    "from cache).",
                    "Stabilize the shared prefix (identical system prompt + tool "
                    "definitions first, dynamic content last) so repeated calls "
                    "hit the cache; provider floors are ~1k-2k tokens.",
                )
            )
        elif hit_ratio >= _GOOD_CACHE_HIT_RATIO:
            findings.append(
                _finding(
                    "cache_efficiency",
                    "info",
                    f"{hit_ratio:.0%} of input tokens were served from cache "
                    f"({cached} of {input_tokens}).",
                    "Cache usage is healthy — keep the stable prefix unchanged "
                    "across calls.",
                )
            )

    return findings


# --------------------------------------------------------------------------
# prompt_size
# --------------------------------------------------------------------------


def _check_prompt_size(usage: dict[str, Any] | None) -> list[dict[str, str]]:
    if not usage:
        return []
    input_tokens = usage.get("input_tokens") or 0
    calls = max(1, usage.get("llm_call_count") or 1)
    per_call = input_tokens / calls
    if per_call >= _LARGE_PROMPT_TOKENS:
        return [
            _finding(
                "prompt_size",
                "warning",
                f"Prompts average ~{per_call:.0f} tokens per LLM call "
                f"({input_tokens} tokens over {calls} calls).",
                "Very large prompts multiply cost on every call. Trim or "
                "distill static instructions, load only the tool schemas each "
                "call needs (~200 tokens per tool), and cache the stable "
                "prefix (research documents 59-90% of per-call token cost "
                "sitting in system prompts).",
            )
        ]

    prompt_chars = usage.get("prompt_chars") or 0
    system_chars = usage.get("system_prompt_chars") or 0
    if prompt_chars > 0 and system_chars / prompt_chars >= _STATIC_DOMINANT_SHARE:
        est_tokens = usage.get("prompt_token_estimate") or per_call
        severity = "warning" if est_tokens >= _STATIC_LARGE_TOKENS else "info"
        return [
            _finding(
                "prompt_size",
                severity,
                f"Static instructions are {system_chars / prompt_chars:.0%} of "
                f"the prompt (~{est_tokens:.0f} tokens by char estimate).",
                "Static instruction text dominates the prompt: make it a stable "
                "cached prefix and consider distilling it — documented "
                "compressions reach 90%+ with minimal task impact, but measure "
                "quality before adopting any compression.",
            )
        ]
    return []


# --------------------------------------------------------------------------
# repeated_instructions
# --------------------------------------------------------------------------


def _check_repeated_instructions(usage: dict[str, Any] | None) -> list[dict[str, str]]:
    if not usage:
        return []
    repeated = usage.get("repeated_prompt_blocks") or 0
    if repeated < 1:
        return []
    calls = usage.get("llm_call_count") or 0
    if calls < 2:
        return []

    input_tokens = usage.get("input_tokens") or 0
    cached = usage.get("cached_input_tokens") or 0
    hit_ratio = cached / input_tokens if input_tokens else 0.0
    severity = "warning" if hit_ratio < 0.5 else "info"
    return [
        _finding(
            "repeated_instructions",
            severity,
            f"The same instruction block(s) were re-sent on {repeated + 1}+ "
            f"occasions across the node's {calls} LLM calls "
            f"(cache hit rate {hit_ratio:.0%}).",
            "Re-sent instruction blocks are the ideal cached prefix: keep them "
            "byte-identical and first in the prompt so every call reuses the "
            "cached KV state instead of re-paying for them.",
        )
    ]


# --------------------------------------------------------------------------
# output_discipline
# --------------------------------------------------------------------------


def _check_output_discipline(
    record: TraceRecord, usage: dict[str, Any] | None
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    if record.status == "error":
        if record.error_type == "retry_loop" and usage:
            spent = usage.get("total_tokens") or 0
            if spent > 0:
                findings.append(
                    _finding(
                        "output_discipline",
                        "warning",
                        f"The node burned {spent} tokens before its rate-limit "
                        "retry loop exhausted — all of it spent for no output "
                        "the graph could use.",
                        "Retry backoff windows are dead cost: reduce per-call "
                        "prompt size to lower request cost per retry, or cap "
                        "retries earlier and fail over to a cheaper model.",
                    )
                )
        return findings

    if record.original_route == "unknown":
        findings.append(
            _finding(
                "output_discipline",
                "warning",
                "The node completed but no route signal was found in its "
                "output (route resolved to 'unknown').",
                "The call's output tokens bought nothing the graph could use. "
                "Constrain the output to a schema (structured output / enum "
                "choice) so a usable route is always produced.",
            )
        )
        return findings

    if usage:
        calls = max(1, usage.get("llm_call_count") or 1)
        output_per_call = (usage.get("output_tokens") or 0) / calls
        if output_per_call >= _VERBOSE_OUTPUT_TOKENS and len(record.original_route) <= _LABEL_MAX_CHARS:
            findings.append(
                _finding(
                    "output_discipline",
                    "warning",
                    f"Output averages ~{output_per_call:.0f} tokens per call "
                    f"for a label-like route ('{record.original_route}').",
                    "An enum-like route should cost near-zero output tokens: "
                    "use constrained/structured output (grammar or choice "
                    "constraint, or a dspy adapter) — also removes "
                    "parse-failure retries, a second token cost.",
                )
            )
    return findings


# --------------------------------------------------------------------------
# observer_overhead
# --------------------------------------------------------------------------


def _check_observer_overhead(
    record: TraceRecord, usage: dict[str, Any] | None
) -> list[dict[str, str]]:
    overhead = record.analysis_overhead
    if not isinstance(overhead, dict):
        return []
    overhead_tokens = overhead.get("total_tokens") or 0
    if overhead_tokens <= 0:
        return []

    node_tokens = (usage or {}).get("total_tokens") or 0
    cost = overhead.get("cost_usd")
    cost_part = f" (~${cost:.4f})" if isinstance(cost, (int, float)) and cost > 0 else ""
    retries = overhead.get("retries") or 0
    retry_part = f", {retries} rate-limit retries" if retries else ""

    severity = "info"
    evidence_extra = ""
    if node_tokens > 0 and overhead_tokens > _OVERHEAD_DOMINANCE * node_tokens:
        severity = "warning"
        evidence_extra = (
            f" — that is {overhead_tokens / node_tokens:.1f}x the node's own "
            f"{node_tokens} tokens"
        )

    return [
        _finding(
            "observer_overhead",
            severity,
            f"Conntrail's analysis overhead for this trace was {overhead_tokens} "
            f"tokens{cost_part}{retry_part}{evidence_extra}.",
            "The observer re-runs the node 4x plus one contrast call per "
            "sampled trace. If overhead dominates, lower sample_rate (each "
            "sampled call pays this; unsampled calls pay nothing).",
        )
    ]
