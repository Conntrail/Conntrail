"""
Live CPE-GEPA optimizer run (G2) — a real, non-mocked run against a live
Anthropic API, optimizing G1's CustomerSupportRouter with entropy-guided
feedback from real Conntrail traces.

Bridging notes (discovered by dry-running this against a real installed
dspy.GEPA — Conntrail-Lib's gepa/*.py was ported having never once been
exercised against real dspy.GEPA, per EPICS.md's own risk callout, and two
real incompatibilities turned up):

1. dspy 3.x's GEPA requires its metric to accept 5 args
   (gold, pred, trace, pred_name, pred_trace) and return either a float or
   a dspy.Prediction(score=, feedback=) — not the 3-arg/tuple contract the
   ported CPEFeedbackFunction originally assumed. Fixed in
   src/conntrail/gepa/feedback.py (also now writes the computed task score
   back onto the attempt record, since end_attempt() is called before the
   score is known).

2. GEPA evaluates trainset examples through the student module itself, and
   deep-copies the student per candidate to keep instruction mutations
   isolated. TraceCollector holds a threading.Lock, which can't be
   deep-copied — dspy's fallback (a *shallow* copy) silently produces a
   second, diverged TraceCollector per candidate that CPEFeedbackFunction
   never sees. The fix (below) is to never store the collector/config as
   instance attributes on the traced dspy.Module at all — keep them in a
   factory closure instead, so dspy's per-instance copying can't touch
   them, while `self.inner` (a genuine dspy submodule) still gets copied
   and mutated correctly per candidate.

Usage:
    # Against a real Anthropic API (the original G2 scope):
    export ANTHROPIC_API_KEY=...
    python examples/gepa/run_live.py --max-metric-calls 8

    # Against a local OpenAI-compatible server (Unsloth Studio, Ollama,
    # llama.cpp, vLLM) — same "local/<name>" convention as providers.py,
    # LOCAL_* env vars for URL/auth:
    python examples/gepa/run_live.py \
        --student-model local/unsloth/gemma-4-12b-it-GGUF \
        --reflection-model local/unsloth/gemma-4-12b-it-GGUF \
        --max-metric-calls 8

Not library code — a runnable example living outside the installable
packages, per EPICS.md Phase 5.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import dspy
import httpx
from dspy.utils.usage_tracker import track_usage

sys.path.insert(0, str(Path(__file__).parent))

from customer_support_student import CustomerSupportRouter  # noqa: E402
from trainset import TRAINSET  # noqa: E402

from conntrail import ConntrailConfig  # noqa: E402
from conntrail.cost import TokenUsage, estimate_cost  # noqa: E402
from conntrail.gepa import CPEGEPAOptimizer  # noqa: E402
from conntrail.gepa.bridge import TraceCollector  # noqa: E402
from conntrail.interceptor import NodeInterceptor  # noqa: E402

logger = logging.getLogger("conntrail.examples.gepa")

DEFAULT_STUDENT_MODEL = "claude-haiku-4-5-20251001"  # matches ConntrailConfig's own default
DEFAULT_REFLECTION_MODEL = "claude-opus-5"
# Reflection on a small local model can't spare 4000 tokens of budget —
# keep prompts/answers short instead.
_LOCAL_REFLECTION_MAX_TOKENS = 1500


def _is_local(model: str) -> bool:
    return model == "local" or model.startswith("local/")


def make_lm(model: str, *, api_key: str | None, max_tokens: int) -> dspy.LM:
    """Build a dspy.LM from a model string.

    "local/<name>" routes to the local OpenAI-compatible server at
    LOCAL_LLM_URL (auth per LOCAL_AUTH_MODE — JWT for Unsloth Studio), with
    chain-of-thought disabled in JWT mode so small max_tokens budgets aren't
    consumed by reasoning. Anything else is an Anthropic model name, per the
    original G2 scope.

    cache=False: dspy's persistent response cache returns cache_hit responses
    with empty usage, which makes every attempt's token stamp (and the
    cost-weighted scoring that reads it) silently None on warm prompts. A
    cost-measuring run must see real per-call usage, so caching is off.
    """
    if _is_local(model):
        from conntrail.utils.providers import _resolve_local_api_key, local_chat_kwargs

        name = model.split("/", 1)[1] if "/" in model else os.environ.get(
            "LOCAL_MODEL_NAME", "local-model"
        )
        return dspy.LM(
            f"openai/{name}",
            api_base=os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8888/v1"),
            api_key=_resolve_local_api_key(),
            max_tokens=max_tokens,
            temperature=0.0,
            cache=False,
            **local_chat_kwargs(),
        )
    if not api_key:
        raise RuntimeError(f"ANTHROPIC_API_KEY is required for non-local model {model!r}.")
    return dspy.LM(
        f"anthropic/{model}", api_key=api_key, max_tokens=max_tokens, cache=False
    )


def _dspy_usage_summary(tracker) -> dict[str, int] | None:
    """Flatten dspy's per-LM usage totals into the attempt token_usage shape."""
    totals = tracker.get_total_tokens()
    if not totals:
        return None
    input_tokens = output_tokens = cached_tokens = 0
    for entry in totals.values():
        input_tokens += int(entry.get("prompt_tokens") or 0)
        output_tokens += int(entry.get("completion_tokens") or 0)
        details = entry.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cached_tokens += int(details.get("cached_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_tokens,
    }


def _dspy_usage_cost(tracker) -> float | None:
    """Estimate USD cost of a tracker's usage via the SDK's price table."""
    total = 0.0
    seen = False
    for lm_name, entry in tracker.get_total_tokens().items():
        details = entry.get("prompt_tokens_details") or {}
        usage = TokenUsage(
            input_tokens=int(entry.get("prompt_tokens") or 0),
            output_tokens=int(entry.get("completion_tokens") or 0),
            cached_input_tokens=int(details.get("cached_tokens") or 0)
            if isinstance(details, dict)
            else 0,
        )
        seen = True
        total += estimate_cost(lm_name, usage)
    return round(total, 6) if seen else None


def make_traced_router_class(
    collector: TraceCollector,
    config: ConntrailConfig,
    *,
    local_student: bool = False,
) -> type[dspy.Module]:
    """Build a traced dspy.Module class bound to `collector`/`config` via closure.

    forward() builds a fresh NodeInterceptor per call, driving it off
    `self.inner` — always the *currently executing* GEPA candidate copy, so
    that whatever instructions GEPA mutated onto that copy are exactly what gets
    traced. `collector`/`config` stay out of instance state entirely (see
    module docstring, point 2) so dspy's per-candidate module copying can't
    silently fork them.

    The student's dspy LM calls are invisible to the SDK's LangChain cost
    capture, so forward() wraps the traced rollout in dspy's own usage
    tracker and stamps the totals (plus estimated cost for cloud models —
    local servers are marginal-cost-free) onto the attempt record. The
    tracker covers the hot-path call AND the analyser's 4 divergence re-runs;
    every rollout pays that same constant, so candidate-vs-baseline cost
    comparisons in the feedback function stay fair.
    """

    class TracedCustomerSupportRouter(dspy.Module):
        def __init__(self, inner: CustomerSupportRouter) -> None:
            super().__init__()
            self.inner = inner  # a real dspy submodule — GEPA copies/mutates it correctly

        def forward(self, message: str) -> dspy.Prediction:
            async def classify_query(state: dict) -> dict:
                prediction = self.inner(message=state["message"])
                return {**state, "category": prediction.category}

            interceptor = NodeInterceptor(
                classify_query,
                node_id="classify_query",
                config=config,
                input_key="message",
                route_key="category",
            )
            prompt_candidate = self.inner.classify.signature.instructions
            collector.begin_attempt(prompt_candidate)
            with track_usage() as tracker:
                state = asyncio.run(interceptor({"message": message, "category": None}))
            attempt = collector.end_attempt()
            attempt.token_usage = _dspy_usage_summary(tracker)
            attempt.cost_usd = (
                0.0 if local_student and attempt.token_usage else _dspy_usage_cost(tracker)
            )
            return dspy.Prediction(category=state["category"])

    return TracedCustomerSupportRouter


def make_collector_poster(collector_url: str, api_key: str | None, run_id: str):
    """Build an on_attempt_scored callback (G3) that POSTs each fully-scored
    attempt to a running collector's /v1/gepa-attempts, synchronously, one
    request per attempt, as it's scored — not batched at the end.
    """
    headers = {"X-API-Key": api_key} if api_key else {}

    def _post(attempt) -> None:
        payload = {
            "run_id": run_id,
            "attempt_id": attempt.attempt_id,
            "prompt_candidate": attempt.prompt_candidate,
            "scalar_score": attempt.scalar_score,
            "traces": [t.to_dict() for t in attempt.traces],
            "token_usage": attempt.token_usage,
            "cost_usd": attempt.total_cost_usd,
            "latency_ms": attempt.mean_latency_ms,
        }
        try:
            response = httpx.post(
                f"{collector_url.rstrip('/')}/v1/gepa-attempts",
                json=payload,
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "conntrail: failed to persist gepa attempt %r to collector: %s",
                attempt.attempt_id,
                exc,
            )

    return _post


def make_task_metric_fn():
    """Task accuracy: 1.0 if the predicted category matches the example's expected category."""

    def _metric(gold: dspy.Example, pred: dspy.Prediction) -> float:
        return 1.0 if pred.category == gold.category else 0.0

    return _metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=8,
        help="Total GEPA metric-call budget (small on purpose — this is a "
             "correctness run, not a full optimization). Default: 8.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=6,
        help=f"How many trainset examples to use (out of {len(TRAINSET)} available). Default: 6.",
    )
    parser.add_argument(
        "--student-model",
        default=os.environ.get("CONNTRAIL_GEPA_STUDENT_MODEL", DEFAULT_STUDENT_MODEL),
        help="Student LM. Cloud model name (Anthropic) or 'local/<name>' for the "
             "local server. Defaults to CONNTRAIL_GEPA_STUDENT_MODEL, then "
             f"{DEFAULT_STUDENT_MODEL}.",
    )
    parser.add_argument(
        "--reflection-model",
        default=os.environ.get("CONNTRAIL_GEPA_REFLECTION_MODEL", DEFAULT_REFLECTION_MODEL),
        help="Reflection LM, same conventions as --student-model. Defaults to "
             f"CONNTRAIL_GEPA_REFLECTION_MODEL, then {DEFAULT_REFLECTION_MODEL}.",
    )
    parser.add_argument(
        "--contrast-model",
        default=os.environ.get("CONNTRAIL_CONTRAST_MODEL", "claude-haiku-4-5-20251001"),
        help="Contrast-generation model used by the tracing itself (never the "
        "student model). Same conventions as --student-model. Defaults to "
        "CONNTRAIL_CONTRAST_MODEL, then the SDK default.",
    )
    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.1,
        help="Lambda folding token cost into the GEPA score: "
        "score = task - cost_weight * (tokens/baseline - 1). 0 disables cost "
        "scoring. Default: 0.1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "run_live_results.json",
        help="Where to write the run summary (attempt records + optimized instructions).",
    )
    parser.add_argument(
        "--collector-url",
        default=os.environ.get("COLLECTOR_URL"),
        help="If set, POST each scored attempt to this collector's /v1/gepa-attempts "
        "as it's scored (G3). Defaults to the COLLECTOR_URL env var; omit both to "
        "skip persistence and just write --output locally.",
    )
    parser.add_argument(
        "--collector-api-key",
        default=os.environ.get("COLLECTOR_API_KEY"),
        help="X-API-Key for --collector-url. Defaults to the COLLECTOR_API_KEY env var.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Load repo-root .env (if present) so LOCAL_*/CONNTRAIL_* conventions work
    # without exporting them first — same behavior as the test suite's conftest.
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parent.parent.parent / ".env")
    except ImportError:
        pass
    args = parse_args()

    # ANTHROPIC_API_KEY is only required when either LM is a cloud model;
    # "local/..." models authenticate against the local server instead.
    api_key: str | None = None
    if not (_is_local(args.student_model) and _is_local(args.reflection_model)):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required for a live CPE-GEPA run with cloud "
                "models. Set it in the environment (or .env), or pass "
                "--student-model/--reflection-model local/<name> to run against "
                "the local server instead."
            )

    student_max_tokens = 100 if _is_local(args.student_model) else 20
    reflection_max_tokens = (
        _LOCAL_REFLECTION_MAX_TOKENS if _is_local(args.reflection_model) else 4000
    )

    dspy.settings.configure(
        lm=make_lm(args.student_model, api_key=api_key, max_tokens=student_max_tokens)
    )

    trainset = TRAINSET[: args.num_examples]
    if len(trainset) < 2:
        raise ValueError("Need at least 2 trainset examples for a GEPA run.")

    run_id = str(uuid.uuid4())
    on_attempt_scored = None
    if args.collector_url:
        on_attempt_scored = make_collector_poster(
            args.collector_url, args.collector_api_key, run_id
        )
        logger.info("Persisting attempts to %s under run_id=%s", args.collector_url, run_id)

    # Two-phase construction: CPEGEPAOptimizer.__init__ is what builds the
    # TraceCollector + wired-on_alert ConntrailConfig that CPEFeedbackFunction
    # actually reads from — the traced student has to be built *against those
    # specific objects*, which don't exist until after __init__ runs. compile()
    # only reads self.student when it's actually called, so a placeholder here
    # is safe.
    optimizer = CPEGEPAOptimizer(
        student=None,
        trainset=trainset,
        task_metric_fn=make_task_metric_fn(),
        base_conntrail_config=ConntrailConfig(
            contrast_model=args.contrast_model,
            sample_rate=1.0,
            entropy_alert_threshold=0.0,
            async_mode=False,
        ),
        on_attempt_scored=on_attempt_scored,
        cost_weight=args.cost_weight,
        gepa_kwargs={
            "max_metric_calls": args.max_metric_calls,
            "reflection_lm": make_lm(
                args.reflection_model, api_key=api_key, max_tokens=reflection_max_tokens
            ),
            # TraceCollector's begin/end-attempt lifecycle assumes one attempt
            # in flight at a time (see module docstring) — GEPA's default
            # concurrent example evaluation races and corrupts it. Sequential
            # evaluation is required until/unless TraceCollector grows
            # per-thread attempt tracking.
            "num_threads": 1,
        },
    )
    TracedRouter = make_traced_router_class(
        optimizer.collector,
        optimizer.conntrail_config,
        local_student=_is_local(args.student_model),
    )
    optimizer.student = TracedRouter(CustomerSupportRouter())

    logger.info(
        "Starting CPE-GEPA compile(): %d trainset examples, max_metric_calls=%d, "
        "student=%s, reflection=%s",
        len(trainset),
        args.max_metric_calls,
        args.student_model,
        args.reflection_model,
    )
    optimized = optimizer.compile()

    attempts = optimizer.attempt_records
    logger.info("compile() finished with %d recorded attempts.", len(attempts))

    summary = {
        "run_id": run_id,
        "num_trainset_examples": len(trainset),
        "max_metric_calls": args.max_metric_calls,
        "cost_weight": args.cost_weight,
        "num_attempts": len(attempts),
        "attempts": [
            {
                "attempt_id": a.attempt_id,
                "prompt_candidate": a.prompt_candidate,
                "scalar_score": a.scalar_score,
                "mean_entropy": a.mean_entropy,
                "fragile_count": a.fragile_count,
                "boundary_count": a.boundary_count,
                "dominant_attribution": a.dominant_attribution,
                "num_traces": len(a.traces),
                "total_input_tokens": a.total_input_tokens,
                "total_output_tokens": a.total_output_tokens,
                "total_cost_usd": a.total_cost_usd,
                "mean_latency_ms": a.mean_latency_ms,
            }
            for a in attempts
        ],
    }

    if attempts:
        # Each attempt is one example's routing decision, not one whole
        # candidate pass — group by the actual prompt text GEPA had active to
        # get an honest per-candidate accuracy comparison.
        by_candidate: dict[str, list] = defaultdict(list)
        for a in attempts:
            by_candidate[a.prompt_candidate].append(a)

        def _mean_score(group: list) -> float | None:
            scores = [a.scalar_score for a in group if a.scalar_score is not None]
            return sum(scores) / len(scores) if scores else None

        def _mean_entropy(group: list) -> float | None:
            entropies = [a.mean_entropy for a in group if a.mean_entropy is not None]
            return sum(entropies) / len(entropies) if entropies else None

        def _mean_tokens(group: list) -> float | None:
            tokens = [
                (a.total_input_tokens or 0) + (a.total_output_tokens or 0)
                for a in group
                if a.total_input_tokens is not None or a.total_output_tokens is not None
            ]
            return sum(tokens) / len(tokens) if tokens else None

        candidates = list(by_candidate.items())
        print(f"\nAttempts recorded: {len(attempts)} across {len(candidates)} distinct prompt candidate(s)")
        for i, (prompt, group) in enumerate(candidates):
            mean_tokens = _mean_tokens(group)
            tokens_part = f", mean tokens: {mean_tokens:.0f}" if mean_tokens is not None else ""
            print(
                f"  candidate {i} ({len(group)} attempts) — "
                f"mean task accuracy: {_mean_score(group)}, mean entropy: {_mean_entropy(group)}"
                f"{tokens_part}"
            )
            print(f"    prompt: {prompt[:150]!r}")

        if len(candidates) > 1:
            first_score, last_score = _mean_score(candidates[0][1]), _mean_score(candidates[-1][1])
            if first_score is not None and last_score is not None:
                delta = last_score - first_score
                print(f"\nMeasured task-accuracy delta (last candidate - first candidate): {delta:+.3f}")
                print(
                    "This is what this run actually measured — it has no relation to "
                    "Conntrail-Lib's old '+10%' figure, which came from a different, "
                    "unrelated bespoke loop."
                )
    else:
        print(
            "\nNo attempts were recorded — check that on_alert fired "
            "(entropy_alert_threshold=0.0)."
        )

    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nRun summary written to {args.output}")
    print(f"Optimized student: {optimized}")


if __name__ == "__main__":
    main()
