"""
CollectorClient — thin async HTTP client over the collector's query API
(T3) and GEPA-attempts API (G3).
"""
from __future__ import annotations

import os
from typing import Any

import httpx


class CollectorClient:
    """Talks to a running Conntrail collector's GET /v1/traces[/...] endpoints.

    Args:
        base_url: Collector base URL. Defaults to the COLLECTOR_URL env var,
            falling back to "http://localhost:8000".
        api_key: Sent as X-API-Key if set. Defaults to the COLLECTOR_API_KEY
            env var (may be None — the collector then must be unauthenticated).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("COLLECTOR_URL", "http://localhost:8000")).rstrip(
            "/"
        )
        self.api_key = api_key if api_key is not None else os.environ.get("COLLECTOR_API_KEY")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    async def list_traces(self, **params: Any) -> dict[str, Any]:
        """GET /v1/traces with the given (None-valued params dropped) query params."""
        query = {k: v for k, v in params.items() if v is not None}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/v1/traces", params=query, headers=self._headers()
            )
        response.raise_for_status()
        return response.json()

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """GET /v1/traces/{trace_id}. Returns None on 404."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/v1/traces/{trace_id}", headers=self._headers()
            )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def get_gepa_attempts(self, run_id: str) -> dict[str, Any] | None:
        """GET /v1/gepa-attempts?run_id=.... Returns None on 404 (unknown run)."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/v1/gepa-attempts",
                params={"run_id": run_id},
                headers=self._headers(),
            )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def get_cost_summary(self) -> dict[str, Any]:
        """GET /v1/cost-summary — per-node aggregated cost telemetry."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/v1/cost-summary", headers=self._headers()
            )
        response.raise_for_status()
        return response.json()
