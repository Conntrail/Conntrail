"""
Conntrail collector — FastAPI app wiring ingest + query routes to a TraceStore.

Run with: uvicorn conntrail_server.app:create_app --factory
(the --factory flag calls create_app() at server startup rather than at
import time, so COLLECTOR_DB_PATH is read fresh and no sqlite file is
created as a side effect of merely importing this module).
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from conntrail_server.auth import warn_if_auth_disabled
from conntrail_server.routes import ingest, query
from conntrail_server.store import TraceStore


def create_app(db_path: str | Path | None = None) -> FastAPI:
    """Build the collector FastAPI app, backed by a TraceStore at db_path.

    db_path defaults to the COLLECTOR_DB_PATH env var, falling back to
    './conntrail_traces.sqlite3' if unset.
    """
    warn_if_auth_disabled()

    app = FastAPI(title="Conntrail Collector")
    resolved_db_path = db_path or os.environ.get("COLLECTOR_DB_PATH", "conntrail_traces.sqlite3")
    app.state.store = TraceStore(resolved_db_path)

    app.include_router(ingest.router)
    app.include_router(query.router)

    return app
