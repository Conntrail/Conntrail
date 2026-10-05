"""
conntrail_client — a small, dependency-light HTTP client for the Conntrail
collector's read API.

Shared by the dashboard and the report generator so the wire contract lives in
exactly one place. Only depends on ``httpx``. The method names and return
shapes mirror ``conntrail_dashboard.client.CollectorClient``.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

__all__ = ["CollectorClient"]


class CollectorClient:
    """Talks to a running Conntrail collector's GET /v1/... endpoints.

    Args:
        base_url: Collector base URL. Defaults to the COLLECTOR_URL env var,
            falling back to "http://localhost:8000".
        api_key: Sent as X-API-Key if set. Defaults to the COLLECTOR_API_KEY
            env var (may be None — the collector then must be unauthenticated).
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("COLLECTOR_URL", "http://localhost:8000")
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("COLLECTOR_API_KEY")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    async def _get(self, path: str, **params: Any) -> httpx.Response:
        query = {k: v for k, v in params.items() if v is not None}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.get(f"{self.base_url}{path}", params=query, headers=self._headers())

    async def list_traces(self, **params: Any) -> dict[str, Any]:
        """GET /v1/traces with the given (None-valued params dropped) query params."""
        response = await self._get("/v1/traces", **params)
        response.raise_for_status()
        return response.json()

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """GET /v1/traces/{trace_id}. Returns None on 404."""
        response = await self._get(f"/v1/traces/{trace_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def get_gepa_attempts(self, run_id: str) -> dict[str, Any] | None:
        """GET /v1/gepa-attempts?run_id=.... Returns None on 404 (unknown run)."""
        response = await self._get("/v1/gepa-attempts", run_id=run_id)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def get_cost_summary(self, **params: Any) -> dict[str, Any]:
        """GET /v1/cost-summary — per-node aggregated cost telemetry."""
        response = await self._get("/v1/cost-summary", **params)
        response.raise_for_status()
        return response.json()

    async def get_export(self, **params: Any) -> dict[str, Any]:
        """GET /v1/export — a bundled cost-summary + trace set for reporting."""
        response = await self._get("/v1/export", **params)
        response.raise_for_status()
        return response.json()
