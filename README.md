# Conntrail

Standalone observability for agentic workflows. Conntrail wraps your agent's
routing decisions, measures how **stable** they are under semantic perturbation
of the input, classifies failures, and ships the records to a collector with a
dashboard — without changing your agent's behavior.

## The idea

Most agent tracing tells you *what* happened. Conntrail tells you *how brittle
the decision was*:

1. Your node runs normally (`trace_node` wraps it; output is unchanged).
2. Conntrail's contrast generator (a cheap LLM, never your agent's model)
   produces three perturbed variants of the input: a **similar** paraphrase, a
   **neutral** urgency-stripped version, and an **opposite** semantic inversion.
3. The analyser re-runs your node with all four inputs and computes
   **normalized Shannon entropy** over the four routing outcomes —
   `0.0` = the route never flips (confident), `1.0` = maximally unstable
   (fragile).
4. A fixed attribution table names which dimension first flipped the route
   (`opposite > neutral > similar`), plus the counterfactual route — "remove
   that dimension and the agent would have taken the *other* path".

Every observation is a `TraceRecord` exported over HTTP to a collector, never
blocking your agent's hot path (`async_mode=True` fires analysis as a
background task). Failures are observed, never swallowed: if your node raises,
Conntrail records an error trace **and re-raises the original exception**.

## Architecture

Four installable units, one repo:

| Package | What it is | Extras |
|---|---|---|
| `conntrail` | The SDK: `trace_node()` / `trace_graph()`, contrast generation, divergence analysis, `HttpExporter` | base (`langchain-core`, `pydantic`, `httpx`) |
| `conntrail_server` | The collector: FastAPI ingest + query API over sqlite | `conntrail[server]` |
| `conntrail_dashboard` | Trace explorer UI: FastAPI + Jinja2 + HTMX (no JS build step) | `conntrail[dashboard]` |
| `conntrail.gepa` | CPE-GEPA optimizer: `dspy.GEPA` with entropy-guided feedback from Conntrail traces | `conntrail[gepa]` |

```
your agent ──trace_node()──► TraceRecord ──HttpExporter──► collector (sqlite)
                                                             ▲
                                    dashboard (HTMX) ────────┘
```

## Install

```bash
pip install -e .                    # SDK only
pip install -e ".[server,dashboard,gepa,dev]"   # everything + test tooling
```

Python ≥ 3.11.

## Quickstart

### 1. Trace a node

`trace_node()` works on any dict-in/dict-out function — a LangGraph node, or
anything else:

```python
from conntrail import ConntrailConfig, trace_node
from conntrail.exporters.http import HttpExporter

config = ConntrailConfig(
    contrast_model="claude-haiku-4-5-20251001",   # cheap model for contrast generation
    exporter=HttpExporter("http://localhost:8000", api_key="..."),
    sample_rate=1.0,          # trace every call (0.1–0.2 for prod)
    async_mode=True,          # never block the hot path
)

@trace_node(config=config, input_key="message", route_key="route")
async def my_router_node(state: dict) -> dict:
    ...
    return {**state, "route": route}
```

For a whole compiled LangGraph graph in one call:

```python
from conntrail import trace_graph

graph = trace_graph(compiled_graph, config=config)   # LangGraph-specific adapter
```

Stability labels: entropy `< 0.25` → **confident**, `0.25–0.60` → **boundary**,
`> 0.60` → **fragile** (fires `on_alert` above `entropy_alert_threshold`).

### 2. Run the collector

```bash
export COLLECTOR_API_KEY=change-me          # static X-API-Key gate (401 without it)
export COLLECTOR_DB_PATH=conntrail_traces.sqlite3
uvicorn conntrail_server.app:create_app --factory --port 8000
```

Endpoints: `POST /v1/traces` (ingest), `GET /v1/traces` (filter by `node_id`,
`stability`, `status`, `failure_category`, `since`/`until`, paginated),
`GET /v1/traces/{id}` (full record), `GET /v1/cost-summary` (per-node token /
cost / cache-hit aggregates + cross-node shared instruction blocks),
`GET /healthz`. Leaving `COLLECTOR_API_KEY` unset runs unauthenticated with a
loud startup warning — dev convenience only.

### 3. Run the dashboard

```bash
export COLLECTOR_URL=http://localhost:8000
export COLLECTOR_API_KEY=change-me
uvicorn conntrail_dashboard.app:create_app --factory --port 8001
```

Pages: trace list (filterable, HTMX partial swaps), per-trace detail
(contrasts, entropy, attribution, counterfactual, cost telemetry + findings),
failure view (grouped by category), a per-node **cost view** (tokens, cache
hit ratio, estimated cost, shared instruction blocks), and a before/after
CPE-GEPA panel (now with token/cost deltas).

### 4. Or just docker compose it

```bash
cp .env.example .env   # fill in COLLECTOR_API_KEY, host ports
docker compose -f deploy/docker-compose.yml --env-file .env up --build
```

Collector (sqlite on a mounted volume, survives restarts) + dashboard, both
healthchecked.

## Failure classification

Every ingested trace is classified server-side from signals the SDK already
produces — no invented categories:

| Category | Signal |
|---|---|
| `exception` | `status="error"`, any uncaught exception from the node |
| `retry_loop` | the analyser's rate-limit retry loop was exhausted (`RetryExhaustedError`) |
| `timeout` | async node exceeded `ConntrailConfig.timeout_seconds` |
| `malformed_output` | `status="ok"` but no route signal found in the output (`original_route == "unknown"`) |
| `none` | a normal, resolved-route success |

## Cost telemetry

Alongside stability, every trace records what the node *cost* (on by default,
`capture_cost=True`): wall-clock latency, the node's internal LLM token usage
observed via a LangChain callback handler (input/output/**cache-read**/cache-
write tokens, per model — best-effort: it sees calls made through LangChain
clients, so non-LangChain nodes report no usage), and **Conntrail's own
analysis overhead** (the 4 re-runs + contrast call — the observer's bill is
measured too, so `sample_rate` can be tuned from data).

`cost_usd` figures come from a built-in price table (longest-prefix match,
`local/*` = free, `CONNTRAIL_PRICE_OVERRIDES` env or
`ConntrailConfig.model_prices` for corrections — prices drift, treat every
figure as an estimate).

Each trace also carries derived **cost findings** (same
derived-not-invented philosophy as the failure classifier):

| Dimension | Answers |
|---|---|
| `cache_efficiency` | is the right stuff being cached? (hit rate, wasted cache writes, missing `cache_control` on Anthropic, sub-threshold prompts) |
| `prompt_size` | are prompts unnecessarily large? (per-call token anomalies, static-instruction dominance) |
| `repeated_instructions` | are identical instruction blocks re-sent across the node's calls without being cache-served? |
| `output_discipline` | are output tokens held to a standard? (verbose free-form output for label-like routes, `malformed_output`/`retry_loop` wasted-token accounting) |
| `observer_overhead` | what Conntrail's own analysis cost for this trace |

Findings recommend, never mutate — research shows naive compression can
crater accuracy (LLMLingua-2 destroys tool schemas; documented cases of 32% →
8% accuracy at 50% token reduction), so the human — or the GEPA reflection
LM — decides what to act on.

The collector aggregates all of this at `GET /v1/cost-summary`, and the
dashboard's **Cost** page shows per-node tokens / cache hit ratio / cost /
latency / warnings, plus **shared instruction blocks across nodes** (the same
instruction fingerprint sent by multiple nodes — the shared cached-prefix
candidate). Trace detail pages show per-trace usage, overhead, and findings.

### Cost in the CPE-GEPA loop

The optimizer scores cost alongside stability and task accuracy:

- `score = task_score − cost_weight × (tokens / baseline_tokens − 1)` — the
  seed prompt pays no penalty, cheaper candidates are rewarded. Default
  `cost_weight=0.1` (`--cost-weight 0` disables). The attempt's
  `scalar_score` stays the *raw* task score for honest reporting.
- `objective_scores={"cost": −tokens}` is also emitted for gepa's native
  per-objective Pareto tracking.
- The feedback text (the only channel the reflection LM reads) reports the
  candidate's tokens/cost and standing cost guidance — concise instructions,
  byte-stable prefixes for caching, schema-constrained outputs. Approach
  validated by CROP (arXiv:2604.14214).

## LLM providers

Contrast generation (and the GEPA student/reflection LMs) resolve through
`conntrail.utils.providers.get_chat_model()` — provider inferred from the
model-name prefix (`claude-*`, `gpt-*`, `gemini-*`, `llama-*`/Groq, …),
falling back to the first available API key. Cloud keys: `GROQ_API_KEY`,
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` (or `GEMINI_API_KEY`),
`OPENROUTER_API_KEY`.

**Local servers** (Unsloth Studio, Ollama, llama.cpp, vLLM — anything
OpenAI-compatible): pass `model="local/<name>"` and configure via env:

```
LOCAL_LLM_URL=http://127.0.0.1:8888/v1
LOCAL_MODEL_NAME=unsloth/gemma-4-12b-it-GGUF
LOCAL_AUTH_MODE=none | api_key | jwt     # jwt = Unsloth Studio username/password exchange
LOCAL_USERNAME=...  LOCAL_PASSWORD=...
```

Cost telemetry works across all of these: Anthropic/Gemini/Groq report token
usage natively; OpenAI, OpenRouter, and the OpenAI-compatible local servers
(Unsloth, Ollama `/v1`, llama.cpp, vLLM) get `stream_usage=True` requested
explicitly, since langchain-openai only auto-enables usage reporting for the
default api.openai.com base URL. In JWT mode the token is re-exchanged on
every call (a server restart never produces stale-401 failures) and
chain-of-thought is disabled so small token budgets aren't consumed by
reasoning. See `.env.example` for every variable.

## CPE-GEPA (prompt optimization from traces)

`examples/gepa/run_live.py` runs a real `dspy.GEPA` optimization whose feedback
function scores each prompt candidate by the routing entropy of its Conntrail
traces (`CPEGEPAOptimizer`). Attempts are POSTed to the collector's
`/v1/gepa-attempts` as they're scored; the dashboard's before/after panel
compares the first vs. last attempt.

```bash
python examples/gepa/run_live.py --max-metric-calls 8 \
    --student-model local/unsloth/gemma-4-12b-it-GGUF \
    --reflection-model local/unsloth/gemma-4-12b-it-GGUF \
    --collector-url http://localhost:8000
```

Note: any accuracy delta this run prints is what *this run* measured. It has
no relation to numbers from other codebases or bespoke loops.

## Testing

```bash
pytest tests/unit -m "not integration"        # pure unit, no LLM, ~3s
pytest tests/integration -m "integration"     # live-LLM round trips
ruff check .
```

Integration tests run against either a cloud key or the local server —
see **docs/TESTING.md** for the full test plan, the local-LLM rig, and
verified results.

## Repo layout

```
src/conntrail/            SDK: interceptor, analyser, contrast, record, wrap, cost, cost_analyzer, exporters, gepa/
src/conntrail_server/     collector: routes (ingest/query/gepa/cost), store, classifier, auth, migrations/
src/conntrail_dashboard/  dashboard: routes, client, templates/, static/
examples/gepa/            G1 student module + trainset + live GEPA run script
deploy/                   Dockerfile.server, Dockerfile.dashboard, docker-compose.yml
docs/                     EPICS.md (work breakdown), TESTING.md (test plan + results)
tests/                    unit/ + integration/ + fixtures/
```

## Status

v1 complete: all six phases of `docs/EPICS.md` are implemented and verified —
tracing core, collector + sqlite store + auth, failure classification,
dashboard, the real CPE-GEPA path, and containerized deploy.
