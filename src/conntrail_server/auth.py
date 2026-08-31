"""
Static shared-secret auth gate for the collector.

Not full multi-tenant auth — a single X-API-Key header checked against the
COLLECTOR_API_KEY env var. If COLLECTOR_API_KEY is unset, the collector runs
unauthenticated (dev convenience) and logs a clear startup warning so this
never silently ships to prod insecure.
"""
from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Request

logger = logging.getLogger("conntrail_server")


def warn_if_auth_disabled() -> None:
    """Log a startup warning if COLLECTOR_API_KEY is unset."""
    if not os.environ.get("COLLECTOR_API_KEY"):
        logger.warning(
            "conntrail_server: COLLECTOR_API_KEY is not set — the collector is running "
            "UNAUTHENTICATED. Set COLLECTOR_API_KEY before deploying anywhere reachable "
            "beyond local development."
        )


async def require_api_key(request: Request) -> None:
    """FastAPI dependency: enforce X-API-Key when COLLECTOR_API_KEY is set.

    With COLLECTOR_API_KEY unset, every request passes through (dev mode).
    """
    expected = os.environ.get("COLLECTOR_API_KEY")
    if not expected:
        return
    provided = request.headers.get("X-API-Key")
    if provided != expected:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
