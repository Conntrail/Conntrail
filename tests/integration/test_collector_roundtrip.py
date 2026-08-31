"""
Integration test: trace_node() -> HttpExporter -> a real running collector
(subprocess uvicorn) -> sqlite -> query API, end to end.

Needs a live LLM key for the happy-path node's contrast generation, same as
F7's integration test (tests/integration/test_trace_graph_live.py).
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from conntrail import ConntrailConfig, trace_node
from conntrail.exporters.http import HttpExporter

_API_KEY = "test-collector-key"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_ready(base_url: str, api_key: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/v1/traces", headers={"X-API-Key": api_key}, timeout=1.0)
            if resp.status_code == 200:
                return
        except httpx.RequestError as exc:
            last_exc = exc
        time.sleep(0.1)
    raise RuntimeError(f"collector did not become ready in time (last error: {last_exc})")


@pytest.mark.integration
class TestCollectorRoundtrip:
    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not any(os.getenv(k) for k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
            pytest.skip("No API key available")

    @pytest.fixture
    def collector(self, tmp_path):
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        db_path = tmp_path / "collector_roundtrip.sqlite3"

        env = {
            **os.environ,
            "COLLECTOR_DB_PATH": str(db_path),
            "COLLECTOR_API_KEY": _API_KEY,
        }
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "conntrail_server.app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_until_ready(base_url, _API_KEY)
            yield base_url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    async def _latest_trace_for_node(self, base_url: str, node_id: str) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{base_url}/v1/traces",
                params={"node_id": node_id, "limit": 1},
                headers={"X-API-Key": _API_KEY},
            )
        resp.raise_for_status()
        traces = resp.json()["traces"]
        assert traces, f"no traces found for node_id={node_id!r}"
        return traces[0]

    @pytest.mark.asyncio
    async def test_happy_path_trace_is_queryable_via_collector(self, collector):
        base_url = collector
        exporter = HttpExporter(collector_url=base_url, api_key=_API_KEY)
        config = ConntrailConfig(exporter=exporter, sample_rate=1.0, async_mode=False)

        @trace_node(config=config)
        async def roundtrip_router(state):
            text = state["message"].lower()
            route = "escalate" if "urgent" in text else "general"
            return {**state, "route": route}

        state = {"message": "This is urgent, please help", "route": None}
        output = await roundtrip_router(state)
        assert output["route"] == "escalate"

        summary = await self._latest_trace_for_node(base_url, "roundtrip_router")
        detail = await self._fetch_detail(base_url, summary["trace_id"])

        assert detail["node_id"] == "roundtrip_router"
        assert detail["status"] == "ok"
        assert detail["original_route"] == "escalate"
        assert 0.0 <= detail["entropy_score"] <= 1.0
        assert detail["stability"] in ("confident", "boundary", "fragile")

        await exporter.close()

    @pytest.mark.asyncio
    async def test_error_path_trace_round_trips_with_error_status(self, collector):
        base_url = collector
        exporter = HttpExporter(collector_url=base_url, api_key=_API_KEY)
        config = ConntrailConfig(exporter=exporter, sample_rate=1.0, async_mode=False)

        @trace_node(config=config)
        async def roundtrip_failing_router(state):
            raise ValueError("simulated node failure")

        with pytest.raises(ValueError, match="simulated node failure"):
            await roundtrip_failing_router({"message": "hello"})

        summary = await self._latest_trace_for_node(base_url, "roundtrip_failing_router")
        detail = await self._fetch_detail(base_url, summary["trace_id"])

        assert detail["status"] == "error"
        assert detail["error_type"] == "ValueError"
        assert detail["error_message"] == "simulated node failure"

        await exporter.close()

    async def _fetch_detail(self, base_url: str, trace_id: str) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{base_url}/v1/traces/{trace_id}",
                headers={"X-API-Key": _API_KEY},
            )
        resp.raise_for_status()
        return resp.json()
