"""
ConntrailConfig — single configuration object passed at setup time.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from conntrail.exporters.base import BaseExporter


@dataclass
class ConntrailConfig:
    """
    Configuration for Conntrail tracing.

    Attributes:
        contrast_model: LLM model ID used for contrast generation.
            Should be a cheap/fast model — never the model being traced.
        sample_rate: Fraction of node calls to analyse [0.0, 1.0].
            1.0 = trace every call (dev default). 0.1-0.2 recommended for prod.
        async_mode: When True, contrast analysis never blocks the hot path.
        exporter: Where to write TraceRecords. Pass a configured BaseExporter
            instance (e.g. HttpExporter) directly — there's no format string
            to dispatch on. Leave unset (None) to run analysis without
            exporting anywhere (e.g. pure-SDK unit testing).
        entropy_alert_threshold: entropy_score >= this value triggers on_alert.
        on_alert: Optional callback fired when a fragile node is detected.
            Signature: (trace_record: TraceRecord) -> None
        timeout_seconds: If set, an async node_fn call exceeding this many
            seconds is cancelled and recorded as an error trace
            (error_type="timeout"); the original asyncio.TimeoutError still
            propagates to the caller. None (default) preserves no-timeout
            behavior. Not enforced for sync node_fn calls.
    """

    contrast_model: str = "claude-haiku-4-5-20251001"
    sample_rate: float = 1.0
    async_mode: bool = True
    exporter: BaseExporter | None = None
    entropy_alert_threshold: float = 0.6
    on_alert: Callable | None = field(default=None, repr=False)
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0.0, 1.0], got {self.sample_rate}")
        if not 0.0 <= self.entropy_alert_threshold <= 1.0:
            raise ValueError(
                f"entropy_alert_threshold must be in [0.0, 1.0], got {self.entropy_alert_threshold}"
            )
