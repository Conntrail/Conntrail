from __future__ import annotations

from collections.abc import Callable

from conntrail.config import ConntrailConfig

from .bridge import TraceCollector
from .feedback import CPEFeedbackFunction


class CPEGEPAOptimizer:
    """
    Wraps dspy.GEPA with CPE-guided feedback from Conntrail traces.

    Example:
        optimizer = CPEGEPAOptimizer(
            student=my_dspy_module,
            trainset=train_examples,
            task_metric_fn=my_accuracy_fn,
            base_conntrail_config=ConntrailConfig(
                sample_rate=1.0,
                entropy_alert_threshold=0.0,
                async_mode=False,
            ),
            gepa_kwargs={"num_iterations": 10},
        )
        optimized = optimizer.compile()

    Requirements:
        - async_mode=False in ConntrailConfig: traces must complete before feedback fires.
        - entropy_alert_threshold=0.0: collect all traces, not only high-entropy ones.
        - sample_rate=1.0: trace every routing decision during optimization.

    cost_weight folds the candidate's token cost into the GEPA score
    (task score - cost_weight * (tokens/baseline - 1)); 0 disables it. The
    base_conntrail_config's capture_cost/model_prices control the underlying
    telemetry.
    """

    def __init__(
        self,
        student,
        trainset: list,
        task_metric_fn: Callable | None = None,
        base_conntrail_config: ConntrailConfig | None = None,
        gepa_kwargs: dict | None = None,
        on_attempt_scored: Callable | None = None,
        cost_weight: float = 0.1,
        valset: list | None = None,
        output_field: str | None = None,
    ) -> None:
        self.student = student
        self.trainset = trainset
        self.valset = valset
        self._task_metric = task_metric_fn
        self._base_config = base_conntrail_config
        self._gepa_kwargs = gepa_kwargs or {}
        self.cost_weight = cost_weight

        self.collector = TraceCollector()
        self.conntrail_config = self.collector.make_config(base_conntrail_config)
        self.feedback_fn = CPEFeedbackFunction(
            self.collector,
            task_metric_fn,
            on_attempt_scored,
            cost_weight=cost_weight,
            output_field=output_field,
        )

    def compile(self):
        try:
            import dspy
        except ImportError as exc:
            raise ImportError(
                "dspy-ai is required for CPEGEPAOptimizer. "
                "Install it with: pip install 'conntrail[gepa]'"
            ) from exc

        gepa = dspy.GEPA(
            metric=self.feedback_fn,
            **self._gepa_kwargs,
        )
        # Pass a held-out valset when given: without one dspy reuses the
        # trainset and the reported score is train-fit, not generalization.
        return gepa.compile(self.student, trainset=self.trainset, valset=self.valset)

    @property
    def attempt_records(self):
        """All PromptAttemptRecords collected during compile()."""
        return self.collector.all_attempts
