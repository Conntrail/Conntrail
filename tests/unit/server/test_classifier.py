"""
Tests for the failure classifier (C2).
"""
from __future__ import annotations

from conntrail_server.classifier import classify_failure


def test_retry_loop_classified_from_error_type():
    record = {"status": "error", "error_type": "retry_loop", "original_route": "unknown"}
    assert classify_failure(record) == "retry_loop"


def test_timeout_classified_from_error_type():
    record = {"status": "error", "error_type": "timeout", "original_route": "unknown"}
    assert classify_failure(record) == "timeout"


def test_generic_exception_classified_as_exception():
    record = {"status": "error", "error_type": "ValueError", "original_route": "unknown"}
    assert classify_failure(record) == "exception"


def test_malformed_output_when_ok_but_route_unknown():
    record = {"status": "ok", "error_type": None, "original_route": "unknown"}
    assert classify_failure(record) == "malformed_output"


def test_none_for_normal_successful_resolved_route():
    record = {"status": "ok", "error_type": None, "original_route": "refund"}
    assert classify_failure(record) == "none"


def test_ok_status_with_resolved_route_is_unambiguously_none_even_with_stale_error_type():
    """A stale/leftover error_type on a status='ok' record must not leak into
    the classification — status='ok' with a real route is always 'none'."""
    record = {"status": "ok", "error_type": "retry_loop", "original_route": "refund"}
    assert classify_failure(record) == "none"


def test_status_defaults_to_ok_when_missing():
    record = {"original_route": "refund"}
    assert classify_failure(record) == "none"
