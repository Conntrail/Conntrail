from __future__ import annotations

from typing import TYPE_CHECKING

from conntrail.cost import estimate_tokens

if TYPE_CHECKING:
    from .bridge import TraceCollector

from .schema import PromptAttemptRecord

_DEFAULT_COST_WEIGHT = 0.1

_STABILITY_ADVICE = {
    "fragile": (
        "Multiple routing decisions changed under semantic paraphrase. "
        "The prompt is underspecified at this decision point — small wording "
        "differences redirect the agent. Consider adding explicit routing criteria "
        "or examples for this case type."
    ),
    "boundary": (
        "Some routing decisions were inconsistent under paraphrase. "
        "The prompt is near a decision boundary. Clarifying language or a concrete "
        "example may stabilise routing."
    ),
    "confident": (
        "Routing was stable under paraphrase. "
        "The prompt is clear at this decision point."
    ),
}

_COST_GUIDANCE = (
    "Cost guidance: prefer concise prompt candidates (shorter instructions cost "
    "less input on every call), keep instructions byte-stable so repeated calls "
    "hit the provider's prompt cache, and elicit short schema-constrained "
    "outputs — a label-like answer should cost near-zero output tokens."
)


def _attempt_tokens(attempt: PromptAttemptRecord) -> int | None:
    """Total in+out tokens for an attempt, or None when no usage data exists."""
    input_tokens, output_tokens = attempt.total_input_tokens, attempt.total_output_tokens
    if input_tokens is None and output_tokens is None:
        return None
    return (input_tokens or 0) + (output_tokens or 0)


def cpe_feedback(attempt: PromptAttemptRecord, baseline: PromptAttemptRecord | None = None) -> str:
    """
    Converts a PromptAttemptRecord into a natural-language feedback string
    suitable for GEPA's textual feedback slot.

    Includes cost telemetry when available (tokens, estimated USD, and a
    comparison against the first attempt's baseline), plus standing cost
    guidance — the feedback string is the only channel GEPA's reflection LM
    reads, so cost advice placed here is what steers prompt candidates
    toward cheaper instructions (validated approach: CROP, arXiv:2604.14214).
    """
    if not attempt.traces:
        return (
            "No Conntrail traces were collected for this prompt attempt. "
            "Cannot provide entropy-based feedback. Check that the agent ran "
            "with Conntrail instrumentation and entropy_alert_threshold=0.0."
        )

    mean_e = attempt.mean_entropy
    fragile = attempt.fragile_count
    boundary = attempt.boundary_count
    confident = len(attempt.traces) - fragile - boundary
    attribution = attempt.dominant_attribution or "unknown dimension"

    if fragile > boundary and fragile > confident:
        dominant = "fragile"
    elif boundary >= fragile and boundary > confident:
        dominant = "boundary"
    else:
        dominant = "confident"

    advice = _STABILITY_ADVICE[dominant]

    lines = [
        f"CPE Analysis ({len(attempt.traces)} routing nodes sampled):",
        f"  Mean entropy: {mean_e:.3f} | "
        f"Fragile: {fragile} | Boundary: {boundary} | Confident: {confident}",
        f"  Primary instability dimension: {attribution}",
        "",
        advice,
    ]

    in_tokens = attempt.total_input_tokens
    out_tokens = attempt.total_output_tokens
    cost = attempt.total_cost_usd
    if in_tokens is not None or out_tokens is not None or cost is not None:
        cost_part = f" (~${cost:.4f})" if cost is not None else ""
        lines.append(
            f"  Cost: {in_tokens or 0} in / {out_tokens or 0} out tokens{cost_part} "
            f"| prompt candidate is ~{estimate_tokens(attempt.prompt_candidate)} tokens."
        )
        baseline_tokens = _attempt_tokens(baseline) if baseline is not None else None
        attempt_tokens = _attempt_tokens(attempt)
        if (
            baseline_tokens
            and attempt_tokens
            and attempt_tokens > baseline_tokens
            and attempt is not baseline
        ):
            lines.append(
                "  This candidate used more tokens than the original prompt — "
                "a shorter candidate that preserves routing quality is preferred."
            )
        lines.append(f"\n{_COST_GUIDANCE}")

    if dominant != "confident" and attempt.scalar_score is not None:
        lines.append(
            f"\nTask score: {attempt.scalar_score:.3f}. "
            "Routing instability likely contributes to score variance across inputs."
        )

    return "\n".join(lines)


class CPEFeedbackFunction:
    """
    Callable wrapper around cpe_feedback for use as GEPA's metric/feedback function.

    dspy 3.x's GEPA metric protocol (GEPAFeedbackMetric) requires a 5-argument
    signature — feedback_fn(gold, pred, trace, pred_name, pred_trace) — and a
    return value of either a plain float or a dspy.Prediction(score=, feedback=)
    (dspy's ScoreWithFeedback shape); a bare (score, feedback) tuple is not
    recognised (dspy's own Prediction iterates over field *names*, not
    values, so a naive tuple-return would silently score wrong). This is a
    deliberate deviation from Conntrail-Lib's 3-arg/tuple-return original —
    verified against a real installed dspy.GEPA, not just the old repo's
    fully-mocked test suite.

    Must be used alongside a TraceCollector. The collector captures traces during
    the rollout; this function retrieves the most recently completed attempt.

    Cost scoring: the returned ``score`` folds cost into the base score as
    ``score = base - cost_weight * (tokens / baseline_tokens - 1)`` where the
    baseline is the first attempt — the seed prompt pays no penalty, cheaper
    candidates are rewarded, costlier ones penalized. GEPA's minibatch
    acceptance is driven by the scalar score, so this actually steers the
    search; ``objective_scores={"cost": -tokens}`` is also emitted for
    gepa's native per-objective Pareto tracking. ``scalar_score`` on the
    attempt record stays the RAW task score (reporting honesty). No usage
    data (non-LangChain student, capture off) means no adjustment.

    Args:
        collector: TraceCollector instance shared with the agent run.
        task_metric_fn: Optional callable(gold, pred) -> float for task accuracy.
                        Falls back to (1 - mean_entropy) as a stability proxy.
        on_attempt_scored: Optional callable(PromptAttemptRecord) invoked once
                        an attempt has both its traces and its real task score
                        (only fires on the task_metric_fn path, since that's
                        the sole point a genuine score exists). Lets a caller
                        (e.g. a live-run script) persist each attempt exactly
                        once, fully formed — not before its score exists, and
                        not more than once.
        cost_weight: Lambda for the cost penalty above (default 0.1: a
                        candidate costing 2x the baseline loses 0.1 score).
                        0 disables cost folding (pure task/stability scoring).
    """

    def __init__(
        self,
        collector: TraceCollector,
        task_metric_fn=None,
        on_attempt_scored=None,
        cost_weight: float = _DEFAULT_COST_WEIGHT,
    ) -> None:
        if cost_weight < 0:
            raise ValueError(f"cost_weight must be >= 0, got {cost_weight}")
        self._collector = collector
        self._task_metric = task_metric_fn
        self._on_attempt_scored = on_attempt_scored
        self._cost_weight = float(cost_weight)

    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
        attempts = self._collector.all_attempts
        if not attempts:
            return self._result(0.0, "No attempts recorded yet.")

        latest = attempts[-1]
        baseline = attempts[0]

        if self._task_metric is not None:
            # Persist onto the record itself — TraceCollector.end_attempt() has no
            # score to give it at call time (the caller doesn't have gold/pred yet),
            # so this is the first point a real task score exists. Without this,
            # every attempt's scalar_score stays None forever and callers reading
            # optimizer.attempt_records after compile() see no accuracy signal.
            # scalar_score keeps the RAW task score; cost folding happens only
            # on the GEPA-facing return value below.
            latest.scalar_score = float(self._task_metric(gold, pred))
            if self._on_attempt_scored is not None:
                self._on_attempt_scored(latest)
            base_score = latest.scalar_score
        elif latest.scalar_score is not None:
            base_score = latest.scalar_score
        elif latest.mean_entropy is not None:
            base_score = 1.0 - latest.mean_entropy
        else:
            base_score = 0.0

        score = self._apply_cost_weight(latest, baseline, base_score)
        feedback = cpe_feedback(latest, baseline=baseline)
        return self._result(score, feedback, attempt=latest)

    def _apply_cost_weight(
        self,
        attempt: PromptAttemptRecord,
        baseline: PromptAttemptRecord,
        base_score: float,
    ) -> float:
        """Fold cost into the scalar so GEPA's acceptance steers toward
        cheaper candidates: score = base - weight * (ratio - 1)."""
        if self._cost_weight <= 0.0:
            return base_score
        ratio = self._cost_ratio(attempt, baseline)
        if ratio is None:
            return base_score
        return base_score - self._cost_weight * (ratio - 1.0)

    @staticmethod
    def _cost_ratio(
        attempt: PromptAttemptRecord, baseline: PromptAttemptRecord
    ) -> float | None:
        """Attempt tokens relative to the baseline (first) attempt; None when
        either side lacks usage data (no cost signal to fold)."""
        attempt_tokens = _attempt_tokens(attempt)
        baseline_tokens = _attempt_tokens(baseline)
        if attempt_tokens is None or not baseline_tokens:
            return None
        return attempt_tokens / baseline_tokens

    @staticmethod
    def _result(score: float, feedback: str, attempt: PromptAttemptRecord | None = None):
        import dspy

        objective_scores = None
        if attempt is not None:
            tokens = _attempt_tokens(attempt)
            if tokens is not None:
                objective_scores = {"cost": -float(tokens)}
        if objective_scores is not None:
            return dspy.Prediction(
                score=score, feedback=feedback, objective_scores=objective_scores
            )
        return dspy.Prediction(score=score, feedback=feedback)
