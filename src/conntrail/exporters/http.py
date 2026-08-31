"""
HttpExporter — POSTs TraceRecords to a running Conntrail collector.

An export failure must never crash the traced host application, mirroring
the "analysis failures never crash the host app" posture elsewhere in
Conntrail: a 5xx or connection error gets one retry, then is logged and
swallowed; a 4xx (validation failure — nothing a retry would fix) is logged
and swallowed immediately, no retry.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from conntrail.exporters.base import BaseExporter
from conntrail.record import TraceRecord

logger = logging.getLogger("conntrail")

_RETRY_BACKOFF_SECONDS = 0.5


class HttpExporter(BaseExporter):
    """
    Exports TraceRecords to a Conntrail collector over HTTP.

    Args:
        collector_url: Base URL of the collector (e.g. "http://localhost:8000").
        api_key: Sent as the X-API-Key header if set. Omitted entirely otherwise.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        collector_url: str,
        api_key: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.collector_url = collector_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def write(self, record: TraceRecord) -> None:
        url = f"{self.collector_url}/v1/traces"
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        payload = record.to_dict()

        for attempt in range(2):  # one initial attempt + one retry
            try:
                response = await self._client.post(url, json=payload, headers=headers)
            except httpx.RequestError as exc:
                if attempt == 0:
                    await self._sleep_before_retry()
                    continue
                logger.warning(
                    "conntrail: export failed for trace %r after retry: %s",
                    record.trace_id,
                    exc,
                )
                return

            if response.status_code < 400:
                return
            if response.status_code >= 500:
                if attempt == 0:
                    await self._sleep_before_retry()
                    continue
                logger.warning(
                    "conntrail: export failed for trace %r after retry: collector "
                    "returned %s",
                    record.trace_id,
                    response.status_code,
                )
                return
            # 4xx — not retryable
            logger.warning(
                "conntrail: export rejected for trace %r: collector returned %s: %s",
                record.trace_id,
                response.status_code,
                response.text,
            )
            return

    async def _sleep_before_retry(self) -> None:
        await asyncio.sleep(_RETRY_BACKOFF_SECONDS)

    async def close(self) -> None:
        await self._client.aclose()
