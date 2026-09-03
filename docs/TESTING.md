# Conntrail — Test Plan & Verified Results

How to test Conntrail end to end: the automated suite, the live-LLM rig, and
the by-hand verification steps the unit suite can't cover (dashboard, GEPA
live run, containers). The "verified" column reflects the last full executed
run of this plan (see §6).

## 1. Automated suite

| Layer | Command | What it covers |
|---|---|---|
| Pure unit | `pytest tests/unit -m "not integration"` (~3 s, no LLM) | parsing edge cases, entropy math, interceptor behavior (incl. F6's record-and-re-raise), store/CRUD, classifier, auth gate, HttpExporter retry logic, dashboard routes vs a mocked collector, GEPA modules with `dspy` mocked |
| Live contrast (optional, F3) | `pytest tests/unit/test_contrast.py -m integration` | `ContrastGenerator.generate()` against a real LLM on fixture inputs |
| Integration | `pytest tests/integration -m "integration or slow"` | F7: `trace_graph` doesn't alter graph output with a live pipeline; T6: full `trace_node → HttpExporter → uvicorn collector → sqlite → query API` round trip (happy + error paths); G2: a real small `CPEGEPAOptimizer.compile()` producing ≥2 attempts with real traces |
| Lint | `ruff check .` | style + import hygiene |

Markers: `integration` (needs a live LLM), `slow` (> 10 s). Both are registered
in `pyproject.toml` and `tests/conftest.py`.

## 2. The live-LLM rig (cloud or local)

Integration tests need *a* live LLM. Two supported configurations:

**Cloud** — set any of `GROQ_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
in `.env`. Tests use the SDK's default contrast model.

**Local server** (Unsloth Studio, Ollama, llama.cpp, vLLM — anything
OpenAI-compatible). In `.env`:

```
LOCAL_LLM_URL=http://127.0.0.1:8888/v1
LOCAL_MODEL_NAME=unsloth/gemma-4-12b-it-GGUF      # must be an installed model
LOCAL_AUTH_MODE=jwt                               # Unsloth Studio username/password
LOCAL_USERNAME=unsloth
LOCAL_PASSWORD=...

CONNTRAIL_CONTRAST_MODEL=local/unsloth/gemma-4-12b-it-GGUF
CONNTRAIL_GEPA_STUDENT_MODEL=local/unsloth/gemma-4-12b-it-GGUF
CONNTRAIL_GEPA_REFLECTION_MODEL=local/unsloth/gemma-4-12b-it-GGUF
```

Notes for the Unsloth Studio rig specifically:

- The server must have the model **loaded** (`POST /v1/load
  {"model_path": "unsloth/gemma-4-12b-it-GGUF"}`; first-ever load also
  downloads the GGUF and can take ~15 min — later loads are ~1 min). List
  installed models with `GET /v1/models`.
- JWT mode disables chain-of-thought (`enable_thinking: false`) — this
  gemma build is a reasoning model and will otherwise spend the whole
  `max_tokens` budget thinking and return an empty answer.
- `langchain-openai` must be installed (included in the `dev` extra) — the
  local/OpenAI-compatible provider path needs it.
- Gates: integration tests skip unless a cloud key **or** an explicit
  `LOCAL_LLM_URL` is present (`tests/conftest.py::live_llm_available`).

## 3. Manual verification plan (by hand, per phase)

These are the steps the automated suite can't exercise. Run them in order
against a scratch sqlite file.

**3.1 Collector + auth (T2–T4)** — start the collector with
`COLLECTOR_API_KEY` set; `curl /healthz` → 200; `/v1/traces` without key →
401; wrong key → 401; correct key → 200. Restart with the key unset →
startup warning in the log, requests pass.

**3.2 GEPA live run + persistence (G2/G3)** — with the collector running:

```bash
python examples/gepa/run_live.py --max-metric-calls 8 \
  --collector-url http://127.0.0.1:8000 --collector-api-key <key>
```

Expect: compile completes, ≥2 attempts recorded, a *real* mean entropy
printed (not `None` — `None` means contrast generation failed, usually a
missing/unreachable contrast model), attempts queryable via
`GET /v1/gepa-attempts?run_id=...`. Report the measured accuracy delta as
what this run measured — never the old repo's "+10%" figure.

**3.3 Dashboard golden path (D1/D2)** — start the dashboard pointed at the
collector; click/filter through: list → filter by stability/failure_category →
per-trace detail (entropy, attribution, counterfactual, contrasts, and for
failed traces error_type/message) → failure view grouped by category →
before/after panel for the run_id from 3.2. HTMX filter swaps return the row
partial only (`HX-Request` header). Unknown trace id → 404 page.

**3.4 External project instrumentation** — wrap a node of a real agent
project with `trace_node()` + `HttpExporter` and confirm real traces land
(see §5 for the Sequence-game harness). Include at least one error path
(node raises → error trace recorded AND exception re-raised to the caller).

**3.5 Containers (O1–O3)** — `docker compose -f deploy/docker-compose.yml
--env-file .env up --build`; both services healthy; auth enforced on the
composed collector; run a small traced function from the host against
`localhost:<COLLECTOR_HOST_PORT>` and see it in the dashboard; `docker
restart` the collector container and confirm the trace survives (sqlite
volume persistence).

## 4. Bugs found while verifying (fixed)

- **`tests/integration/test_gepa_live_run.py` was missing entirely** — G2's
  test plan in EPICS.md requires it. Written and passing (see §6).
- **Dashboard container couldn't start** — `conntrail[dashboard]` didn't
  include `uvicorn`, but `Dockerfile.dashboard` runs via uvicorn (`exec:
  "uvicorn": executable file not found`). Added `uvicorn>=0.30.0` to the
  `dashboard` extra.
- **Local/OpenAI-compatible provider path unrunnable in dev** —
  `langchain-openai` wasn't in any extra. Added to `dev`.
- **`examples/gepa/run_live.py` hardcoded Anthropic** and ignored
  `CONNTRAIL_CONTRAST_MODEL` — contrast generation silently failed with
  local-only setups (attempts recorded with `mean entropy: None`). It now
  supports `local/<name>` models for student/reflection/contrast and loads
  `.env`.

## 5. External test-project harness (Sequence game)

The tournament repo (`AI-Agents-Sequence-Game-Tournament`) is used read-only.
Its agent's full board prompt is too large for contrast generation, so the
harness traces the agent's *strategic routing decision*: each real game
position is distilled (via the game's own analysis functions) into a compact
situation summary, the agent's LLM — through the project's own provider
plumbing and its verbatim strategy system prompt — answers which priority it
follows (`win/block/fork/advance/center/disrupt`), and that node is wrapped
in `trace_node()`. Result: Conntrail measures whether the agent's strategic
route flips under perturbed situation descriptions, plus an error-path trace
(node raises mid-decision). The harness lives outside both repos (nothing in
the tournament project is modified); re-create from §5 of this doc or ask
for the script.

## 6. Verified results (last full run)

Environment: Python 3.14.7, local Unsloth Studio server with
`unsloth/gemma-4-12b-it-GGUF` (UD-Q4_K_XL) as the only live LLM, RTX 4060.

| Check | Result |
|---|---|
| `ruff check .` | clean |
| Pure unit suite | **309 passed** |
| Live contrast tests (F3) | 6 passed, 1 skipped (see §7) |
| F7 `trace_graph` live | passed |
| T6 collector round trip (happy + error) | passed |
| G2 GEPA live compile | passed (~50 s, ≥2 attempts, real entropy) |
| 3.1 collector auth | 401 / 401 / 200 verified live |
| 3.2 GEPA live run + G3 persistence | 9 attempts, mean entropy 0.281, queryable by run_id |
| 3.3 dashboard golden path | list/filter/detail/failure/before-after all render; HTMX partials; 404 on unknown id |
| 3.4 Sequence-game harness | 3 traced positions (route=center, entropy 0.406 boundary, attribution=semantic intensity, counterfactual=advance) + 1 error trace classified `exception` and re-raised |
| 3.5 containers | both images build; compose healthy; auth enforced; host-traced sample visible in dashboard; sqlite survives `docker restart` |

## 7. Known limitations

- **Degenerate inputs on small local models** — the contrast fixture `"x"`
  (single character) makes gemma-4-12b ask for clarification instead of
  emitting JSON; cloud models comply. Skipped locally with an explicit
  reason; not a Conntrail defect (the parse failure path itself is unit-tested).
- **dspy + JWT** — dspy LMs hold one token for the whole run; a
  sufficiently-long GEPA run against Unsloth Studio could outlive the token
  TTL (the LangChain path re-authenticates per call, dspy can't). Small
  correctness runs are unaffected.
- **Failure-view counts** are sampled (`limit`-based, no count endpoint) —
  exact up to 1000 rows per category.
- **`timeout_seconds`** is enforced for async node functions only (a sync
  node_fn can't be cancelled by `asyncio.wait_for`).
