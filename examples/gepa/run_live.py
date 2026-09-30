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
from dataclasses import dataclass
from pathlib import Path

import dspy
import httpx
from dspy.utils.usage_tracker import track_usage

sys.path.insert(0, str(Path(__file__).parent))

from customer_support_student import CustomerSupportRouter  # noqa: E402
from policy_student import PolicyRouter  # noqa: E402
from policy_trainset import POLICY_TRAINSET  # noqa: E402
from trainset import TRAINSET  # noqa: E402

from conntrail import ConntrailConfig  # noqa: E402
from conntrail.cost import TokenUsage, estimate_cost  # noqa: E402
from conntrail.gepa import CPEGEPAOptimizer  # noqa: E402
from conntrail.gepa.bridge import TraceCollector  # noqa: E402
from conntrail.interceptor import NodeInterceptor  # noqa: E402
from conntrail.utils.providers import DEFAULT_MODEL  # noqa: E402

logger = logging.getLogger("conntrail.examples.gepa")

DEFAULT_STUDENT_MODEL = DEFAULT_MODEL
DEFAULT_REFLECTION_MODEL = DEFAULT_MODEL
# Reflection on a small local model can't spare 4000 tokens of budget —
# keep prompts/answers short instead.
_LOCAL_REFLECTION_MAX_TOKENS = 1500

# Demo mode: a plausible-but-weak first draft the optimizer must improve on.
# With the default (already-good) signature instructions GEPA often accepts the
# seed prompt immediately and never mutates, which makes the before/after panel
# flat. Starting here gives the loop something real to fix. Kept category-shaped
# (not "write a reply") so the model's output stays parseable.
WEAK_SEED_INSTRUCTIONS = (
    "Classify the customer message into one category. Default to 'general', "
    "and only answer 'refund' when the customer literally uses the word refund."
)

# Policy-task weak seed: no rules at all, so the model has to guess the
# non-obvious policy (the good default in policy_student.py states it).
POLICY_WEAK_SEED = "Choose the best resolution action for this customer support case."


@dataclass(frozen=True)
class _TaskSpec:
    """How the traced GEPA wrapper drives a particular student task."""

    node_id: str
    student_cls: type
    trainset: list
    weak_seed: str
    input_key: str        # the example field fed to the student
    output_field: str     # the Prediction field holding the decision
    predict_attr: str     # the Predict submodule GEPA mutates


_TASKS: dict[str, _TaskSpec] = {
    # Four-way classification — saturated for strong models (kept for the
    # machinery tests); the policy task below has real headroom.
    "classification": _TaskSpec(
        node_id="classify_query",
        student_cls=CustomerSupportRouter,
        trainset=TRAINSET,
        weak_seed=WEAK_SEED_INSTRUCTIONS,
        input_key="message",
        output_field="category",
        predict_attr="classify",
    ),
    # Policy-following resolution — instruction quality materially changes
    # accuracy, so GEPA has something real to optimize.
    "policy": _TaskSpec(
        node_id="resolve_case",
        student_cls=PolicyRouter,
        trainset=POLICY_TRAINSET,
        weak_seed=POLICY_WEAK_SEED,
        input_key="case",
        output_field="resolution",
        predict_attr="resolve",
    ),
}


def _is_local(model: str) -> bool:
    return model == "local" or model.startswith("local/")


def _is_openrouter(model: str) -> bool:
    return model == "openrouter" or model.startswith("openrouter/")


def _required_key_env(model: str) -> str | None:
    """Env var holding the credential for a model string (None if keyless/local)."""
    if _is_local(model):
        return None
    if _is_openrouter(model):
        return "OPENROUTER_API_KEY"
    return "ANTHROPIC_API_KEY"


def make_lm(model: str, *, max_tokens: int) -> dspy.LM:
    """Build a dspy.LM from a model string.

    "local/<name>" routes to the local OpenAI-compatible server at
    LOCAL_LLM_URL (auth per LOCAL_AUTH_MODE — JWT for Unsloth Studio), with
    chain-of-thought disabled in JWT mode so small max_tokens budgets aren't
    consumed by reasoning. "openrouter/<vendor>/<model>" routes through
    OpenRouter (litellm-native; OPENROUTER_API_KEY). Anything else is an
    Anthropic model name.

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
    if _is_openrouter(model):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"OPENROUTER_API_KEY is required for OpenRouter model {model!r}."
            )
        return dspy.LM(
            model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=0.0,
            cache=False,
            # Skip chain-of-thought: DeepSeek-class reasoning models otherwise
            # think before every rollout/reflection, which roughly triples GEPA
            # wall-clock for a 4-way classification that doesn't need it.
            extra_body={"reasoning": {"enabled": False}},
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(f"ANTHROPIC_API_KEY is required for model {model!r}.")
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
    task: _TaskSpec,
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

    class TracedRouter(dspy.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner  # a real dspy submodule — GEPA copies/mutates it correctly

        def forward(self, **inputs) -> dspy.Prediction:
            text = inputs[task.input_key]

            async def decision(state: dict) -> dict:
                prediction = self.inner(**{task.input_key: state["value"]})
                return {**state, "value": getattr(prediction, task.output_field)}

            interceptor = NodeInterceptor(
                decision,
                node_id=task.node_id,
                config=config,
                input_key="value",
                route_key="value",
            )
            prompt_candidate = getattr(self.inner, task.predict_attr).signature.instructions
            collector.begin_attempt(prompt_candidate)
            with track_usage() as tracker:
                state = asyncio.run(interceptor({"value": text}))
            attempt = collector.end_attempt()
            attempt.token_usage = _dspy_usage_summary(tracker)
            attempt.cost_usd = (
                0.0 if local_student and attempt.token_usage else _dspy_usage_cost(tracker)
            )
            return dspy.Prediction(**{task.output_field: state["value"]})

    return TracedRouter


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


def make_task_metric_fn(task: _TaskSpec):
    """Task accuracy: 1.0 if the predicted decision matches the example's label."""

    def _metric(gold: dspy.Example, pred: dspy.Prediction) -> float:
        return 1.0 if getattr(pred, task.output_field) == getattr(gold, task.output_field) else 0.0

    return _metric


def heldout_accuracy(student, examples, metric_fn, input_key: str) -> float | None:
    """Accuracy on examples the optimizer never trained on.

    A perfect score on the *training* set is a red flag (GEPA can fit a prompt
    to the sample); this is the honest generalization signal. `student` is the
    un-traced dspy module (the traced wrapper's `.inner`), so evaluation adds no
    Conntrail traces and no cost telemetry of its own.
    """
    if not examples:
        return None
    correct = 0.0
    for example in examples:
        try:
            pred = student(**{input_key: getattr(example, input_key)})
        except Exception:  # noqa: BLE001 - a failed rollout counts as wrong
            continue
        correct += metric_fn(example, pred)
    return correct / len(examples)


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
        help="How many trainset examples to use (capped by the selected task's "
        "trainset size). Default: 6.",
    )
    parser.add_argument(
        "--student-model",
        default=os.environ.get("CONNTRAIL_GEPA_STUDENT_MODEL", DEFAULT_STUDENT_MODEL),
        help="Student LM: an OpenRouter slug ('openrouter/<vendor>/<model>'), an "
             "Anthropic model name, or 'local/<name>' for the local server. "
             "Defaults to CONNTRAIL_GEPA_STUDENT_MODEL, then "
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
        default=os.environ.get("CONNTRAIL_CONTRAST_MODEL", DEFAULT_MODEL),
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
        "--task",
        choices=sorted(_TASKS),
        default="classification",
        help="Which student task to optimize: 'classification' (four-way "
        "category — saturated for strong models) or 'policy' (rule-following "
        "resolution — has real headroom). Default: classification.",
    )
    parser.add_argument(
        "--holdout",
        type=int,
        default=3,
        help="Examples held out of optimization as a valset, so the reported "
        "score is generalization rather than train-fit. Set 0 to let GEPA reuse "
        "the trainset (its score is then a training score — 1.0 is a red flag). "
        "Default: 3.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Fraction of rollouts to trace (contrast + 4 divergence re-runs). "
        "Keep at 1.0 when using --cost-weight > 0: a traced rollout makes ~6 LLM "
        "calls vs 1 untraced, so a lower rate makes the token cost signal noise "
        "instead of prompt cost. Lower only if you don't score cost. Default: 1.0.",
    )
    parser.add_argument(
        "--weak-seed",
        action="store_true",
        help="Demo mode: start from a deliberately weak student prompt so GEPA "
        "has something to improve (the default seed prompt already scores high, "
        "so before/after panels are usually flat).",
    )
    parser.add_argument(
        "--seed-prompt",
        default=None,
        help="Override the student's initial instructions with this text. "
        "Takes precedence over --weak-seed.",
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

    # Validate credentials for whichever provider each model needs: local
    # servers use LOCAL_* auth, OpenRouter uses OPENROUTER_API_KEY, everything
    # else is Anthropic (ANTHROPIC_API_KEY). The contrast model is resolved by
    # Conntrail's own provider layer, so validate it too.
    missing: set[str] = set()
    for model in (args.student_model, args.reflection_model, args.contrast_model):
        env_name = _required_key_env(model)
        if env_name and not os.environ.get(env_name):
            missing.add(env_name)
    if missing:
        raise RuntimeError(
            f"{', '.join(sorted(missing))} required for the configured models. "
            "Set the key(s) in the environment (or .env), or pass local/<name> / "
            "openrouter/<vendor>/<model> model strings instead."
        )

    # Cloud students get headroom beyond the bare category label: reasoning
    # models think first, and weak-seed runs write prose until the prompt is
    # fixed — a tiny cap only truncates (and retries) without helping the score.
    student_max_tokens = 100 if _is_local(args.student_model) else 600
    reflection_max_tokens = (
        _LOCAL_REFLECTION_MAX_TOKENS if _is_local(args.reflection_model) else 4000
    )

    dspy.settings.configure(
        lm=make_lm(args.student_model, max_tokens=student_max_tokens)
    )

    task = _TASKS[args.task]
    examples = task.trainset[: args.num_examples]
    holdout = min(args.holdout, max(0, len(examples) - 2))
    if holdout > 0:
        valset = examples[-holdout:]
        trainset = examples[:-holdout]
    else:
        valset = None
        trainset = examples
    if len(trainset) < 2:
        raise ValueError("Need at least 2 trainset examples for a GEPA run.")

    task_metric = make_task_metric_fn(task)

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
        valset=valset,
        task_metric_fn=task_metric,
        output_field=task.output_field,
        base_conntrail_config=ConntrailConfig(
            contrast_model=args.contrast_model,
            sample_rate=args.sample_rate,
            entropy_alert_threshold=0.0,
            async_mode=False,
        ),
        on_attempt_scored=on_attempt_scored,
        cost_weight=args.cost_weight,
        gepa_kwargs={
            "max_metric_calls": args.max_metric_calls,
            "reflection_lm": make_lm(
                args.reflection_model, max_tokens=reflection_max_tokens
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
        task=task,
        local_student=_is_local(args.student_model),
    )
    seed_instructions = args.seed_prompt or (task.weak_seed if args.weak_seed else None)
    if seed_instructions:
        logger.info("Seeding student with custom instructions (%d chars)", len(seed_instructions))
    optimizer.student = TracedRouter(task.student_cls(instructions=seed_instructions))

    logger.info(
        "Starting CPE-GEPA compile(): %d trainset examples (+%d held out), "
        "max_metric_calls=%d, student=%s, reflection=%s",
        len(trainset),
        len(valset) if valset else 0,
        args.max_metric_calls,
        args.student_model,
        args.reflection_model,
    )
    optimized = optimizer.compile()

    seed_heldout = heldout_accuracy(optimizer.student.inner, valset, task_metric, task.input_key)
    optimized_heldout = heldout_accuracy(optimized.inner, valset, task_metric, task.input_key)
    if valset:
        optimized_text = (
            f"{optimized_heldout:.3f}" if optimized_heldout is not None else "n/a"
        )
        print(
            f"\nHeld-out accuracy ({len(valset)} examples never optimized on): "
            f"seed {seed_heldout:.3f} -> optimized {optimized_text}"
        )
    else:
        print(
            "\nNo held-out valset — scores below are training scores, not "
            "generalization (pass --holdout N for an honest number)."
        )

    attempts = optimizer.attempt_records
    logger.info("compile() finished with %d recorded attempts.", len(attempts))

    summary = {
        "run_id": run_id,
        "task": args.task,
        "num_trainset_examples": len(trainset),
        "num_holdout_examples": len(valset) if valset else 0,
        "heldout_accuracy_seed": seed_heldout,
        "heldout_accuracy_optimized": optimized_heldout,
        "max_metric_calls": args.max_metric_calls,
        "cost_weight": args.cost_weight,
        "seed_instructions": seed_instructions,
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
