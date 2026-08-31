"""
Conntrail dashboard — FastAPI + Jinja2 + HTMX trace explorer.

Run with: uvicorn conntrail_dashboard.app:create_app --factory
(the --factory flag calls create_app() at server startup rather than at
import time, so COLLECTOR_URL/COLLECTOR_API_KEY are read fresh).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from conntrail_dashboard.client import CollectorClient
from conntrail_dashboard.routes import router

_BASE_DIR = Path(__file__).parent


def create_app(collector_client: CollectorClient | None = None) -> FastAPI:
    """Build the dashboard FastAPI app.

    collector_client defaults to a CollectorClient() reading COLLECTOR_URL /
    COLLECTOR_API_KEY from the environment. Pass one explicitly for testing.
    """
    app = FastAPI(title="Conntrail Dashboard")
    app.state.collector = collector_client or CollectorClient()
    app.state.templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))

    static_dir = _BASE_DIR / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
