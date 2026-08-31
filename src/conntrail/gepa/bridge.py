from __future__ import annotations

import threading
import uuid

from conntrail.config import ConntrailConfig
from conntrail.record import TraceRecord

from .schema import PromptAttemptRecord


class TraceCollector:
    """
    Thread-safe collector that associates Conntrail TraceRecords with the
    currently active GEPA prompt attempt.

    Usage:
        collector = TraceCollector()
        config = collector.make_config(base_config)

        record = collector.begin_attempt(prompt_candidate_str)
        # run the agent
        completed = collector.end_attempt(scalar_score=0.85)

    Important: pass a ConntrailConfig with async_mode=False and
    entropy_alert_threshold=0.0 so every trace is collected synchronously
    before the feedback function is called.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: PromptAttemptRecord | None = None
        self._completed: list[PromptAttemptRecord] = []

    def make_config(self, base_config: ConntrailConfig | None = None) -> ConntrailConfig:
        """
        Returns a ConntrailConfig that routes trace alerts into this collector.
        Preserves any existing on_alert callback on base_config.
        """
        original_alert = getattr(base_config, "on_alert", None) if base_config else None

        def _on_alert(trace: TraceRecord) -> None:
            with self._lock:
                if self._current is not None:
                    self._current.traces.append(trace)
            if original_alert:
                original_alert(trace)

        kwargs: dict = {}
        if base_config:
            for f in base_config.__dataclass_fields__:
                if f != "on_alert":
                    kwargs[f] = getattr(base_config, f)

        return ConntrailConfig(**kwargs, on_alert=_on_alert)

    def begin_attempt(self, prompt_candidate: str) -> PromptAttemptRecord:
        with self._lock:
            self._current = PromptAttemptRecord(
                attempt_id=str(uuid.uuid4()),
                prompt_candidate=prompt_candidate,
            )
        return self._current

    def end_attempt(self, scalar_score: float | None = None) -> PromptAttemptRecord:
        with self._lock:
            if self._current is None:
                raise RuntimeError("end_attempt called without a matching begin_attempt")
            self._current.scalar_score = scalar_score
            completed = self._current
            self._completed.append(completed)
            self._current = None
        return completed

    @property
    def all_attempts(self) -> list[PromptAttemptRecord]:
        with self._lock:
            return list(self._completed)
