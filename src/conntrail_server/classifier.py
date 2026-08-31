"""
Failure classifier — sorts a trace into one of a small set of categories
using signals Phase 1/C1 already produce (status, error_type) plus
DivergenceAnalyser's "unknown" route fallback. No invented categories —
each one is grounded in a signal that actually exists in the ported code.
"""
from __future__ import annotations

from typing import Any, Literal

FailureCategory = Literal["exception", "retry_loop", "timeout", "malformed_output", "none"]


def classify_failure(record: dict[str, Any]) -> FailureCategory:
    """Classify a TraceRecord.to_dict()-shaped payload into a failure category.

    - retry_loop: status="error", error_type == "retry_loop" (F5/F6's
      RetryExhaustedError tagging — a rate-limit retry loop was exhausted).
    - timeout: status="error", error_type == "timeout" (C1's async node_fn
      timeout via asyncio.wait_for).
    - exception: status="error" and neither of the above — any other
      uncaught exception from node_fn.
    - malformed_output: status="ok" but original_route == "unknown" — the
      analyser's _extract_route strategy-4 fallback fired on an otherwise
      successful call (no route signal found in the output).
    - none: status="ok" with a resolved route — a normal successful trace.
    """
    status = record.get("status", "ok")
    error_type = record.get("error_type")

    if status == "error":
        if error_type == "retry_loop":
            return "retry_loop"
        if error_type == "timeout":
            return "timeout"
        return "exception"

    if record.get("original_route") == "unknown":
        return "malformed_output"

    return "none"
