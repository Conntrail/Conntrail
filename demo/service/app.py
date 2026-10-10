"""
FastAPI app for the Conntrail live demo.

POST /api/analyse  {message, provider?, api_key?, turnstile_token?}
  -> a TraceRecord-shaped analysis of one routing decision.
GET  /api/config   -> which providers/models the server offers.
GET  /healthz      -> liveness + whether a trial key is configured.

The free trial uses the server's own free-tier key and is capped per-IP and
per-day; a visitor may instead send their own free key (bring-your-own-key).
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from demo.service.limits import DemoGuard
from demo.service.models import (
    FREE_MODELS,
    NoProviderKey,
    candidate_configs,
    infer_provider_from_key,
    server_config,
)
from demo.service.runner import run_analysis

MAX_MESSAGE_CHARS = 600

app = FastAPI(title="Conntrail Live Demo")

_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "https://conntrail.ibzie.dev,http://localhost:8090,http://127.0.0.1:8090",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)

guard = DemoGuard()


class AnalyseRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    provider: str | None = None
    api_key: str | None = None
    turnstile_token: str | None = None


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _verify_turnstile(token: str | None, ip: str) -> bool:
    secret = os.getenv("TURNSTILE_SECRET")
    if not secret:
        return True
    if not token:
        return False
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data={"secret": secret, "response": token, "remoteip": ip},
        )
    return bool(resp.json().get("success"))


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "conntrail-demo", "docs": "/api/config"}


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "trial_key_configured": server_config() is not None}


@app.get("/api/config")
async def config() -> dict[str, Any]:
    cfg = server_config()
    return {
        "providers": list(FREE_MODELS),
        "default_provider": cfg.provider if cfg else None,
        "default_model": cfg.model if cfg else None,
        "byok": True,
        "turnstile_site_key": os.getenv("TURNSTILE_SITE_KEY"),
        "daily_remaining": guard.daily_remaining,
        "max_message_chars": MAX_MESSAGE_CHARS,
    }


@app.post("/api/analyse")
async def analyse(payload: AnalyseRequest, request: Request) -> JSONResponse:
    message = payload.message.strip()
    if not message:
        return JSONResponse({"error": "Message cannot be empty."}, status_code=400)
    if len(message) > MAX_MESSAGE_CHARS:
        return JSONResponse({"error": f"Message too long (max {MAX_MESSAGE_CHARS})."}, status_code=400)

    ip = _client_ip(request)
    if not await _verify_turnstile(payload.turnstile_token, ip):
        return JSONResponse({"error": "Bot verification failed."}, status_code=403)

    byok = bool(payload.api_key)
    try:
        configs = candidate_configs(payload.provider, payload.api_key)
    except NoProviderKey as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    # Free trial (server key) is capped; BYOK is the visitor's own quota.
    if not byok:
        allowed, reason = guard.check(ip)
        if not allowed:
            if reason == "daily":
                msg = "The free demo has hit today's limit. Bring your own free key to keep going."
            else:
                msg = "You're analysing quickly — give it a moment, or use your own free key."
            return JSONResponse({"error": msg, "reason": reason}, status_code=429)

    primary = configs[0]
    cache_key = guard.key(message, primary.provider, primary.model)
    cached = guard.cache_get(cache_key)
    if cached is not None:
        return JSONResponse({**cached, "cached": True, "daily_remaining": guard.daily_remaining})

    try:
        result = await run_analysis(message, configs)
    except Exception as exc:  # provider/network errors: keep the demo responsive
        return JSONResponse(
            {"error": f"Analysis failed ({type(exc).__name__}); all free models were busy. Try again shortly."},
            status_code=502,
        )

    guard.cache_set(cache_key, result)
    return JSONResponse({**result, "cached": False, "daily_remaining": guard.daily_remaining})


# Expose the BYOK provider inference so the frontend can prefill it if desired.
def _provider_for_key(api_key: str) -> str | None:  # pragma: no cover - convenience
    return infer_provider_from_key(api_key)
