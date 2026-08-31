from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bridge import TraceCollector

from .schema import PromptAttemptRecord

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


def cpe_feedback(attempt: PromptAttemptRecord) -> str:
    """
    Converts a PromptAttemptRecord into a natural-language feedback string
    suitable for GEPA's textual feedback slot.
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
    """

    def __init__(
        self, collector: TraceCollector, task_metric_fn=None, on_attempt_scored=None
    ) -> None:
        self._collector = collector
        self._task_metric = task_metric_fn
        self._on_attempt_scored = on_attempt_scored

    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
        attempts = self._collector.all_attempts
        if not attempts:
            return self._result(0.0, "No attempts recorded yet.")

        latest = attempts[-1]

        if self._task_metric is not None:
            score = float(self._task_metric(gold, pred))
            # Persist onto the record itself — TraceCollector.end_attempt() has no
            # score to give it at call time (the caller doesn't have gold/pred yet),
            # so this is the first point a real task score exists. Without this,
            # every attempt's scalar_score stays None forever and callers reading
            # optimizer.attempt_records after compile() see no accuracy signal.
            latest.scalar_score = score
            if self._on_attempt_scored is not None:
                self._on_attempt_scored(latest)
        elif latest.scalar_score is not None:
            score = latest.scalar_score
        elif latest.mean_entropy is not None:
            score = 1.0 - latest.mean_entropy
        else:
            score = 0.0

        feedback = cpe_feedback(latest)
        return self._result(score, feedback)

    @staticmethod
    def _result(score: float, feedback: str):
        import dspy

        return dspy.Prediction(score=score, feedback=feedback)
