"""
Tests for HttpExporter (T5).

Uses httpx.MockTransport (part of httpx itself, no extra test dependency)
swapped onto the exporter's internal client, so no real network call happens.
"""
from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from conntrail.contrast import ContrastSet
from conntrail.exporters.http import HttpExporter
from conntrail.record import TraceRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record() -> TraceRecord:
    return TraceRecord(
        trace_id=TraceRecord.make_id(),
        node_id="router",
        timestamp=datetime.now(UTC),
        original_input="I need a refund",
        original_route="refund",
        entropy_score=0.1,
        stability="confident",
        attribution_dimension="urgency",
        plain_language_summary="summary",
        raw_contrasts=ContrastSet(similar="a", neutral="b", opposite="c"),
        raw_outputs={"similar": "refund"},
    )


class RecordingHandler:
    """Canned httpx.MockTransport handler that records every request it sees."""

    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def call_count(self) -> int:
        return len(self.requests)


def _wire_transport(exporter: HttpExporter, handler: RecordingHandler) -> None:
    exporter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retries use a fixed backoff sleep — skip the real wait in tests."""

    async def _instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("conntrail.exporters.http.asyncio.sleep", _instant_sleep)


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


async def test_successful_post_does_not_raise():
    handler = RecordingHandler([httpx.Response(201, json={"trace_id": "x"})])
    exporter = HttpExporter("http://collector.local")
    _wire_transport(exporter, handler)

    await exporter.write(_make_record())

    assert handler.call_count == 1


# ---------------------------------------------------------------------------
# 5xx handling
# ---------------------------------------------------------------------------


async def test_5xx_then_success_retries_once():
    handler = RecordingHandler(
        [httpx.Response(503), httpx.Response(201, json={"trace_id": "x"})]
    )
    exporter = HttpExporter("http://collector.local")
    _wire_transport(exporter, handler)

    await exporter.write(_make_record())

    assert handler.call_count == 2


async def test_5xx_exhausted_does_not_raise(caplog):
    handler = RecordingHandler([httpx.Response(500), httpx.Response(500)])
    exporter = HttpExporter("http://collector.local")
    _wire_transport(exporter, handler)

    await exporter.write(_make_record())  # must not raise

    assert handler.call_count == 2


# ---------------------------------------------------------------------------
# Connection error handling
# ---------------------------------------------------------------------------


async def test_connection_error_then_success_retries_once():
    handler = RecordingHandler(
        [httpx.ConnectError("boom"), httpx.Response(201, json={"trace_id": "x"})]
    )
    exporter = HttpExporter("http://collector.local")
    _wire_transport(exporter, handler)

    await exporter.write(_make_record())

    assert handler.call_count == 2


async def test_connection_error_exhausted_does_not_raise():
    handler = RecordingHandler([httpx.ConnectError("boom"), httpx.ConnectError("boom")])
    exporter = HttpExporter("http://collector.local")
    _wire_transport(exporter, handler)

    await exporter.write(_make_record())  # must not raise

    assert handler.call_count == 2


# ---------------------------------------------------------------------------
# 4xx handling — no retry
# ---------------------------------------------------------------------------


async def test_4xx_does_not_retry_and_does_not_raise():
    handler = RecordingHandler([httpx.Response(422, json={"detail": "bad"})])
    exporter = HttpExporter("http://collector.local")
    _wire_transport(exporter, handler)

    await exporter.write(_make_record())  # must not raise

    assert handler.call_count == 1


# ---------------------------------------------------------------------------
# X-API-Key header
# ---------------------------------------------------------------------------


async def test_api_key_header_present_when_configured():
    handler = RecordingHandler([httpx.Response(201, json={"trace_id": "x"})])
    exporter = HttpExporter("http://collector.local", api_key="secret-123")
    _wire_transport(exporter, handler)

    await exporter.write(_make_record())

    assert handler.requests[0].headers["X-API-Key"] == "secret-123"


async def test_api_key_header_absent_when_not_configured():
    handler = RecordingHandler([httpx.Response(201, json={"trace_id": "x"})])
    exporter = HttpExporter("http://collector.local")
    _wire_transport(exporter, handler)

    await exporter.write(_make_record())

    assert "X-API-Key" not in handler.requests[0].headers
