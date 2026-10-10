# Conntrail live demo service

A small FastAPI service that runs the **real Conntrail SDK** — contrast
generation + divergence analysis — on a visitor's support message, using
free-tier LLM providers. It powers the "Live demo" page on the website.

```
POST /api/analyse  {message, provider?, api_key?, turnstile_token?}
GET  /api/config   -> providers/models the server offers
GET  /healthz      -> liveness + whether a trial key is configured
```

## Why it exists

The static demo on the site is precomputed. This service lets a visitor see
Conntrail *actually run* — 1 contrast call + 4 node re-runs — without us
incurring per-token cost. Two modes:

- **Free trial** — the server's own free-tier key, capped per-IP and per-day,
  with response caching. Cost: $0 as long as the caps hold.
- **Bring-your-own-key** — the visitor supplies a free OpenRouter/Groq key;
  it is sent only to the provider for that request and never stored.

## Free models

Only free-tier providers are supported on purpose. Defaults live in
`models.py` (`FREE_MODELS`) with an ordered fallback list, because free
models are frequently rate-limited (429) or return unusable output:

| Provider | Default model | Notes |
|---|---|---|
| OpenRouter | `cohere/north-mini-code:free` | fallbacks: `poolside/laguna-s-2.1:free`, `dots-studio/dots-3-note-preview:free`, `google/gemma-4-31b-it:free` |
| Groq | `llama-3.3-70b-versatile` | highest published daily cap (1000 req/day, 200K tokens/day) |

Avoid reasoning free models for this task — they spend the token budget
thinking instead of emitting the contrast JSON.

## Run locally

```bash
# in the repo root, with a key in .env (gitignored)
set -a; . ./.env; set +a
uvicorn demo.service.app:app --host 127.0.0.1 --port 8000
```

Then preview the site with `python demo/site/dev_server.py` (port 8090) and open
`/live.html` — it calls the local service automatically.

## Deploy to Render (free plan)

`demo/render.yaml` is a Render blueprint. Render only auto-detects a blueprint
at the **repo root**, so either create the web service manually in the Render
dashboard with the commands below, or copy `demo/render.yaml` to the repo root
when using Blueprints.

- build: `pip install -e ".[server,openai,groq]"`
- start: `uvicorn demo.service.app:app --host 0.0.0.0 --port $PORT`
- health check: `/healthz`

Set the trial key and caps in the Render dashboard (they are `sync: false`,
so they are never read from the repo):

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` *or* `OPENROUTER_API_KEY` | the free trial key |
| `DEMO_PROVIDER` / `DEMO_MODEL_OPENROUTER` | override the default free model |
| `DEMO_PER_IP_LIMIT` | analyses per IP per 10 minutes (default 3) |
| `DEMO_DAILY_CAP` | global analyses per UTC day (default 200) |
| `ALLOWED_ORIGINS` | comma-separated CORS origins |
| `TURNSTILE_SECRET` / `TURNSTILE_SITE_KEY` | optional bot check |

Then set the deployed URL in `demo/site/config.js` (`CONNTRAIL_DEMO_API`).

Render's free plan sleeps after 15 minutes idle (~1 min cold start) and runs a
single instance, so the in-memory rate limiter and cache are sufficient. They
reset on restart, which is fine for a demo.

## Tests

`tests/integration/test_demo_service_live.py` runs against real free models
and is skipped unless a provider key is present. There are no mocked LLM
tests — the point is to verify the real path.
