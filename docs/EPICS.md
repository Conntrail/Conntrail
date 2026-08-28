# Conntrail — Work Breakdown (v1)

Source: `Conntrail.md` §4 (Target Definition) and §7 (Build Backlog), both re-verified against
`Conntrail-Lib/conntrail/` source before this breakdown was written. Every ticket below cites the
actual file(s) it ports from and, where relevant, the exact line(s) driving the ticket.

This is a planning document only — nothing here has been implemented yet. Review before any
ticket is started.

## 0. Decisions this breakdown assumes

These were open questions in `Conntrail.md` §10, resolved with the user before scoping:

| Question | Decision | Effect on this breakdown |
|---|---|---|
| Trace store | **SQLite for v1** | Phase 2 store tickets target sqlite directly, no abstraction layer for a second engine. |
| CPE-GEPA path | **Real `dspy.GEPA` run now** | Phase 5 is scoped as a genuine optimizer run against live data, not a relabel of the bespoke loop. This is the riskiest phase — see G1 note. |
| Export transport | **`HttpExporter` only** | `JsonlExporter`/`StdoutExporter` are **not ported**. `ConntrailConfig.export_format` is dropped entirely; exporters are injected directly (see F6). |
| Auth | **Static API-key gate on the collector** | Phase 2 gets a dedicated auth ticket (T4); not full multi-tenant auth, just a shared-secret header check. |

Additional implementation decisions made while scoping (not asked about explicitly, but needed to
write concrete tickets — flag if you want any of these changed):

- **Package layout:** a single repo, three installable units via `pyproject.toml` extras
  (mirrors the existing `Conntrail-Lib` extras pattern):
  - `conntrail` (base install) — the SDK/tracing core, minimal deps (`langchain-core`, `pydantic`, `httpx`).
  - `conntrail[server]` — collector service (adds `fastapi`, `uvicorn`).
  - `conntrail[dashboard]` — dashboard service (adds `fastapi`, `jinja2`).
  - `conntrail[gepa]` — unchanged from the old repo (`dspy-ai`, `gepa`).
- **Dashboard stack:** FastAPI + Jinja2 + HTMX (+ a CDN chart lib for entropy/before-after visuals),
  not a separate JS/React toolchain. Keeps the self-hosted deploy story (§4/§6) to two plain Python
  containers with no Node build step. Reconsider if you want a richer SPA.
- **Exporter wiring:** `ConntrailConfig` gets a new `exporter: BaseExporter | None` field (a direct
  object, not a format string). This replaces `export_format`/`export_path`/the langsmith branch
  entirely — there's only one real exporter now, so a string dispatcher adds nothing.
- **`docs/` cleanup decisions from §7**, folded into Phase 1 tickets instead of tracked separately:
  - LangSmith exporter stub (`exporters/langsmith.py`, `raise NotImplementedError("Phase 7")`) →
    **dropped**, not ported. Redundant now that HttpExporter is the only transport.
  - Dead embedding-divergence code (`analyser.py`'s `COSINE_THRESHOLD` + unused docstring claim,
    `utils/embedding.py.embed_text()` `NotImplementedError`) → **dropped**, not ported. Dict-diff
    routing extraction is the only mechanism; F5 removes the docstring's false claim instead of
    implementing the stub.
  - Fixed-lookup attribution (`_infer_attribution()`, actually a fixed `opposite > neutral > similar`
    table, not inference) → **kept, relabeled for honesty**. F5 ports the logic unchanged but fixes
    the name/docstring so it stops implying semantic inference.
  - Hardcoded password in `run_local_experiment.sh`, mismarked integration test — these are
    `Conntrail-Lib`-only problems (a script and a test that don't get carried over). No fix ticket
    needed in the old repo; the new repo's own integration suite (F1, T6) is written correctly from
    the start.

---

## Phase 1 — Foundation

Repo skeleton, then port the tracing core. Nothing else can start until this phase is merged.

### F1 — Repo skeleton
**Description:** Set up the new repo's package structure, `pyproject.toml` (base + `server`/`dashboard`/`gepa`/`dev` extras, mirroring `Conntrail-Lib/pyproject.toml`'s extras pattern), and pytest scaffolding. No ported source yet.
**Files touched:** `pyproject.toml`, `src/conntrail/__init__.py`, `src/conntrail_server/__init__.py`, `src/conntrail_dashboard/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`, `tests/conftest.py`, `.env.example`, `ruff` config (`[tool.ruff]` block, same `line-length = 100`, `target-version = "py311"` as the old repo).
**Acceptance criteria:**
- `pip install -e .` succeeds; `pip install -e ".[server,dashboard,gepa,dev]"` succeeds.
- `pytest` runs (0 tests, exit 0) with `testpaths = ["tests"]` and markers `integration` and `slow` registered (drop the old repo's `baseline` marker — that was experiment-specific and experiments aren't being ported).
- `ruff check .` runs clean on the empty skeleton.
**Test plan:** No unit tests (scaffolding only). Verify manually: `pytest --collect-only` succeeds, `pip install -e .` in a clean venv succeeds.
**Dependencies:** none.

### F2 — Port `routing_entropy`
**Description:** Port `conntrail/utils/entropy.py` verbatim — it's correct, pure, and fully covered by the audit as real. `routing_entropy()` computes normalized Shannon entropy over route labels.
**Files touched:** `src/conntrail/utils/entropy.py`.
**Acceptance criteria:** Byte-for-byte equivalent behavior to `Conntrail-Lib/conntrail/utils/entropy.py`; `routing_entropy(["a","a","a","a"]) == 0.0`, `routing_entropy(["a","b","c","d"]) == 1.0`.
**Test plan:** `tests/unit/test_entropy.py` — port the existing test cases (all-same → 0.0, all-different → 1.0, empty list → 0.0, mixed counts, single-element list). No integration test needed (pure function).
**Dependencies:** F1.

### F3 — Port `ContrastGenerator` + provider resolution
**Description:** Port `conntrail/contrast.py` (`ContrastGenerator`, `ContrastSet`, `ContrastGenerationError`) and `conntrail/utils/providers.py` (`get_chat_model`) verbatim, plus `conntrail/prompts/contrast_gen.txt`. Both are real, working code; port as-is including the local-Unsloth-server JWT auth path in `providers.py` (niche but functioning, not a stub).
**Files touched:** `src/conntrail/contrast.py`, `src/conntrail/utils/providers.py`, `src/conntrail/prompts/contrast_gen.txt`.
**Acceptance criteria:** `ContrastGenerator.generate()` produces a `ContrastSet` from a mocked LLM response identically to the old repo's behavior, including markdown-fence stripping, truncated-JSON recovery, and nested-dict value extraction. `get_chat_model()` resolves provider-by-prefix and falls back to the first available API key exactly as before.
**Test plan:** `tests/unit/test_contrast.py` — port `test_contrast.py`'s parsing edge cases (fenced JSON, truncated JSON, nested dict values, missing key, empty input → `ContrastGenerationError`). `tests/unit/test_providers.py` — prefix inference, fallback-provider selection, `local/` routing (mock `urllib` for the token exchange). No integration test for parsing logic; a `@pytest.mark.integration` test hitting a real LLM for `generate()` end-to-end is optional, not required (parsing is the risk surface, not the LLM call itself).
**Dependencies:** F1.

### F4 — Port `TraceRecord`, extend with failure fields
**Description:** Port `conntrail/record.py` (`TraceRecord`, `stability_label`, `build_summary`, `to_dict`/`from_dict`) and extend the dataclass with three new optional fields needed by F6's interceptor fix and Phase 3's classifier: `status: Literal["ok","error"] = "ok"`, `error_type: str | None = None`, `error_message: str | None = None`. Update `to_dict`/`from_dict` to round-trip the new fields.
**Files touched:** `src/conntrail/record.py`.
**Acceptance criteria:** Existing `to_dict()`/`from_dict()` round-trip still exact for all original fields; new fields default to `"ok"`/`None`/`None` so a "success" trace serializes identically in shape to before (three extra keys). `stability_label()` boundaries unchanged (`<0.25` confident, `0.25–0.60` boundary, `>0.60` fragile).
**Test plan:** `tests/unit/test_record.py` — port existing `to_dict`/`from_dict` round-trip tests, `stability_label` boundary tests, `build_summary` string-shape tests; add new tests for `status="error"` round-trip and default values on a normal success record. No integration test (pure data class).
**Dependencies:** F1, F3 (needs `ContrastSet`).

### F5 — Port `DivergenceAnalyser`, drop dead embedding path, fix attribution honesty
**Description:** Port `conntrail/analyser.py`'s `DivergenceAnalyser.analyse()`, `_call_node()` (with its rate-limit retry/backoff), and `_extract_route()` verbatim — this logic is real and already generic (no LangGraph type in the path). Two changes from the original: (1) drop the unused `COSINE_THRESHOLD` constant and the docstring's embedding-divergence claim — don't port `utils/embedding.py` at all; (2) rename `_infer_attribution()`'s framing — keep the fixed `opposite > neutral > similar` lookup logic unchanged, but rewrite the docstring/comment to state plainly it's a fixed priority table, not inference (per §0's honesty decision). Tag the specific case where `_call_node()`'s retry loop exhausts `max_retries` and re-raises with a distinguishable exception so F6/C2 can classify it as `retry_loop` rather than a generic exception (e.g. wrap the final raise in a small `RetryExhaustedError(Exception)` subclass carrying the original error).
**Files touched:** `src/conntrail/analyser.py`. (No `utils/embedding.py` — intentionally not created.)
**Acceptance criteria:** `analyse()` output (`AnalysisResult`) is identical in shape/values to the old repo for the same inputs. `_extract_route()`'s "unknown" fallback (strategy 4) is unchanged and still reachable — this is the signal C2's classifier uses for `malformed_output`. A rate-limit retry loop that exhausts all 4 attempts raises `RetryExhaustedError` instead of a bare `Exception`.
**Test plan:** `tests/unit/test_analyser.py` — port existing tests (route extraction strategies 1–4, entropy computation, attribution priority order, concurrent execution via `asyncio.gather`, rate-limit retry with `retry-after` parsing). Add: retry-exhaustion raises `RetryExhaustedError`; `_extract_route` "unknown" case surfaces correctly when no route signal exists. No integration test (mocked `node_fn`/LLM throughout).
**Dependencies:** F2 (uses `routing_entropy`), F3 (`ContrastSet` type).

### F6 — Port `ConntrailConfig` + `NodeInterceptor`, fix the try/except gap
**Description:** This is the critical ticket. Port `conntrail/config.py` and `conntrail/interceptor.py`, with two deliberate changes: (1) redesign `ConntrailConfig` per §0 — drop `export_format`/`export_path`, add `exporter: BaseExporter | None = None` (direct injection); (2) **fix `NodeInterceptor.__call__`'s try/except gap** (`Conntrail-Lib/conntrail/interceptor.py:70-73`) by wrapping the traced call itself (`self.node_fn(state)` / `await self.node_fn(state)`) in try/except. On exception: build a `TraceRecord` with `status="error"`, `error_type=type(exc).__name__` (or `"retry_loop"` if the exception is F5's `RetryExhaustedError`), `error_message=str(exc)`, `stability="fragile"`, `entropy_score=1.0`, skip the contrast-analysis pipeline (there's no successful output to contrast against), still export the record and fire `on_alert` if configured, then **re-raise** the original exception so the host app's own error handling still runs (Conntrail must observe failures, not swallow them). Port `BaseExporter` (`exporters/base.py`) unchanged as the injection target.
**Files touched:** `src/conntrail/config.py`, `src/conntrail/interceptor.py`, `src/conntrail/exporters/base.py`.
**Acceptance criteria:**
- A `node_fn` that raises is still recorded as a `TraceRecord` (status="error", error_type/error_message populated) *and* the original exception still propagates to the caller.
- A `node_fn` that returns normally behaves identically to the old repo (contrast analysis fires, hot path not blocked when `async_mode=True`).
- `_should_sample()`, `_extract_input_text()`, `_extract_from_messages()` ported unchanged.
- `ConntrailConfig()` with no `exporter` set performs analysis but skips the export step (logs at debug level) rather than erroring — useful for pure-SDK unit testing without a collector running.
**Test plan:** `tests/unit/test_interceptor.py` — port existing suite (output-unchanged, zero-sample-rate, hot-path-not-blocked, sync-node-support, input-extraction fallback/empty-state, on-alert threshold). **New, required tests:** `test_node_exception_is_recorded_and_reraised` (node_fn raises → exporter receives an error-status record AND the exception still propagates out of `__call__`), `test_node_exception_skips_contrast_analysis` (no `ContrastGenerator`/LLM calls happen on the error path — assert via mock), `test_retry_exhausted_classified_distinctly` (a `RetryExhaustedError` from F5 produces `error_type="retry_loop"`). Use a recording test-double `BaseExporter` (not a real HTTP call) throughout — this is exactly the interceptor-decoupling benefit of F6's exporter-injection redesign. No integration test needed (all node_fn/exporter behavior is mockable).
**Dependencies:** F2, F3, F4, F5.

### F7 — Port `trace_node()` / `trace_graph()`
**Description:** Port `conntrail/wrap.py` verbatim — `trace_node()` (generic decorator) and `trace_graph()` (LangGraph-specific, monkeypatches a compiled graph's node callables). Update module/function docstrings to stop implying LangGraph-only usage for `trace_node()`/direct `NodeInterceptor` use (per §4 — "docs need to stop implying LangGraph-only"); `trace_graph()`'s docstring should explicitly say it's the LangGraph-specific convenience adapter, with `trace_node()`/`NodeInterceptor` as the generic path for anything else.
**Files touched:** `src/conntrail/wrap.py`.
**Acceptance criteria:** `trace_node()` and `trace_graph()` behave identically to the old repo (node wrapping, `_active_trace_list` ContextVar handling, `ainvoke`/`invoke` patching for trace collection into `__conntrail_traces__`). Docstrings updated per above; no behavior change from the doc edit.
**Test plan:** `tests/unit/test_wrap.py` — port existing public-API tests (`trace_node` preserves output/name, `trace_graph` wraps all non-system nodes, `_SKIP_NODES` excludes `__start__`/`__end__`). **One integration test**, correctly marked this time (fixing the old repo's mismarking): `tests/integration/test_trace_graph_live.py::test_trace_graph_does_not_alter_graph_output` — needs a live LLM key, marked `@pytest.mark.integration` from the start.
**Dependencies:** F6.

---

## Phase 2 — Transport + Storage

Biggest phase; split by test surface as requested. T1 (store) has no dependency on Phase 1 internals beyond the `TraceRecord` shape (F4), so it can start in parallel with F5–F7 if you want to parallelize within the phase — everything else in Phase 2 depends on T1.

### T1 — Trace store (sqlite schema + repository)
**Description:** New persistent store (none exists today — this is 100% new code, no port). `src/conntrail_server/store.py`: a thin repository over sqlite (stdlib `sqlite3`, no ORM) with `insert(record_dict) -> trace_id`, `get(trace_id) -> dict | None`, `list(filters, limit, offset) -> list[dict]` (filters: `node_id`, `stability`, `status`, `since`/`until` timestamp range). `migrations/0001_init.sql` creates the `traces` table with columns mirroring `TraceRecord.to_dict()` (F4's extended shape) — `raw_contrasts` and `raw_outputs` stored as JSON text columns. Indices on `node_id`, `timestamp`, `stability`, `status`.
**Files touched:** `src/conntrail_server/store.py`, `src/conntrail_server/migrations/0001_init.sql`, `src/conntrail_server/migrations/__init__.py` (simple sequential-file migration runner — no need for Alembic at this scale).
**Acceptance criteria:** `insert()` then `get()` round-trips a full `TraceRecord.to_dict()` payload exactly (including nested `raw_contrasts`/`raw_outputs` JSON). `list()` correctly filters and paginates. Migration runner is idempotent (running it twice doesn't error or duplicate schema).
**Test plan:** `tests/unit/server/test_store.py` — insert/get round-trip, filter by each field independently and combined, pagination (`limit`/`offset`), empty-result cases, migration idempotency. Use a tmp-file or `:memory:` sqlite db per test. No integration test (store has no network surface).
**Dependencies:** F4 (trace record shape), F1.

### T2 — Collector ingest endpoint
**Description:** New FastAPI app (`src/conntrail_server/app.py`) with `POST /v1/traces` (`src/conntrail_server/routes/ingest.py`). Accepts the JSON shape from `TraceRecord.to_dict()` (F4), validates it via a Pydantic request model (`src/conntrail_server/models.py`), writes it to T1's store, returns `201` with `{"trace_id": ...}`. Malformed payloads return `422` with field-level errors (Pydantic default).
**Files touched:** `src/conntrail_server/app.py`, `src/conntrail_server/routes/ingest.py`, `src/conntrail_server/models.py`.
**Acceptance criteria:** Valid `TraceRecord.to_dict()` payload → `201`, row appears in store via `get()`. Missing required field → `422`. Extra unknown fields don't error (forward-compatible).
**Test plan:** `tests/unit/server/test_ingest_route.py` using FastAPI's `TestClient` against an in-memory/tmp store — valid payload accepted, missing-field rejected, malformed JSON rejected, response shape correct. No integration test at this ticket (T6 covers the full round trip).
**Dependencies:** T1.

### T3 — Collector query API
**Description:** `GET /v1/traces` (list with query-param filters: `node_id`, `stability`, `status`, `since`, `until`, `limit`, `offset`) and `GET /v1/traces/{trace_id}` (full record including `raw_contrasts`/`raw_outputs`), in `src/conntrail_server/routes/query.py`. Reads via T1's store.
**Files touched:** `src/conntrail_server/routes/query.py`.
**Acceptance criteria:** List endpoint returns paginated, filtered results matching store semantics. Detail endpoint returns `404` for an unknown `trace_id`, full record otherwise. Response schemas are the dashboard's actual data contract (Phase 4 depends on this exactly).
**Test plan:** `tests/unit/server/test_query_route.py` — list with each filter combination, pagination, 404 on unknown id, full-detail shape includes nested contrast/output JSON correctly deserialized (not double-encoded strings). No integration test at this ticket.
**Dependencies:** T1.

### T4 — API key auth gate
**Description:** Static shared-secret auth for the collector (per §0's decision — not full multi-tenant auth). A FastAPI dependency (`src/conntrail_server/auth.py`) checking an `X-API-Key` header against `COLLECTOR_API_KEY` env var; applied to both `/v1/traces` routes (ingest and query). Missing/wrong key → `401`. If `COLLECTOR_API_KEY` is unset, log a startup warning and run unauthenticated (dev convenience) — document this clearly, don't silently default to an insecure prod posture without a visible warning.
**Files touched:** `src/conntrail_server/auth.py`, `src/conntrail_server/app.py` (wire dependency into routers), `src/conntrail_server/routes/ingest.py`, `src/conntrail_server/routes/query.py`.
**Acceptance criteria:** With `COLLECTOR_API_KEY` set: requests without/with-wrong `X-API-Key` → `401`; correct key → normal response. With it unset: requests succeed unauthenticated, startup log emits a clear warning.
**Test plan:** `tests/unit/server/test_auth.py` — valid key passes, missing key rejected, wrong key rejected, unset-env-var dev-mode passthrough + warning log captured (`caplog`). No integration test needed.
**Dependencies:** T2, T3.

### T5 — `HttpExporter`
**Description:** New `src/conntrail/exporters/http.py::HttpExporter(BaseExporter)`. `write(record)` POSTs `record.to_dict()` (F4's shape) to `f"{collector_url}/v1/traces"` via `httpx.AsyncClient`, sending `X-API-Key` header if `api_key` is set. One retry on transient failure (connection error / 5xx) with a short fixed backoff; a 4xx (validation failure) is not retried and logs a warning rather than raising (mirrors the old repo's "analysis failures never crash the host app" posture — an export failure shouldn't crash the traced application either). Constructor: `HttpExporter(collector_url: str, api_key: str | None = None, timeout: float = 5.0)`.
**Files touched:** `src/conntrail/exporters/http.py`.
**Acceptance criteria:** Successful POST → no error. Server 5xx or connection error → one retry, then logs and returns without raising. Server 4xx → logs and returns without raising (no retry). `X-API-Key` header present iff `api_key` is set.
**Test plan:** `tests/unit/test_http_exporter.py` — mock `httpx.AsyncClient` (e.g. `respx` or a monkeypatched transport): success case, 5xx-then-success retry, 5xx-exhausted-no-raise, 4xx-no-retry-no-raise, header presence/absence. No integration test at this ticket (T6 covers it against a real running collector).
**Dependencies:** T2 (wire contract), F6 (`BaseExporter` interface).

### T6 — Integration: full ingestion round trip
**Description:** The integration test explicitly called out as missing in `Conntrail.md` §"Testing" — nothing in `Conntrail-Lib` covers this because it never had a collector. Exercises `trace_node()` (F7) → `HttpExporter` (T5) → running collector (T2/T3/T4) → sqlite store (T1) → query API (T3), end to end, against a real (locally spun-up) collector process.
**Files touched:** `tests/integration/test_collector_roundtrip.py`.
**Acceptance criteria:** A `trace_node()`-wrapped function, called with `HttpExporter` pointed at a locally started collector instance, results in a trace queryable via `GET /v1/traces/{trace_id}` with correct entropy/stability/route data. Test spins up the collector (e.g. via `TestClient` against the real app + a tmp sqlite file, or a subprocess `uvicorn` if `TestClient` can't exercise the real network path convincingly) and tears it down cleanly.
**Test plan:** One `@pytest.mark.integration` test (needs a live LLM key for the traced node's contrast generation, same as F7's integration test) covering the happy path; a second case for the error path (traced node raises → error-status record round-trips through the same pipeline, per F6/T2).
**Dependencies:** T1, T2, T3, T4, T5, F6, F7.

---

## Phase 3 — Failure Classification

Depends on Phase 1 only (per the requested phase ordering) for the raw signal; C2 additionally touches the collector (T2) to persist the classification, so sequence C2 after Phase 2's T1/T2 are merged even though the classification *logic* itself only needs Phase 1's data.

### C1 — Timeout wrapping
**Description:** New capability — no timeout concept exists in Conntrail today. Add `timeout_seconds: float | None = None` to `ConntrailConfig` (F6). In `NodeInterceptor.__call__`, wrap the traced call in `asyncio.wait_for(..., timeout=self.config.timeout_seconds)` when set. On timeout: same error-record path as F6's exception handling, with `error_type="timeout"`, then re-raise `asyncio.TimeoutError`.
**Files touched:** `src/conntrail/config.py`, `src/conntrail/interceptor.py`.
**Acceptance criteria:** A `node_fn` that exceeds `timeout_seconds` is recorded with `status="error"`, `error_type="timeout"`, and `asyncio.TimeoutError` propagates to the caller. `timeout_seconds=None` (default) preserves current no-timeout behavior exactly.
**Test plan:** `tests/unit/test_interceptor.py` (extends F6's suite) — `test_node_timeout_is_recorded_and_reraised` using an `asyncio.sleep`-based slow `node_fn` and a short `timeout_seconds`. No integration test needed.
**Dependencies:** F6.

### C2 — Failure classifier module
**Description:** New module, `src/conntrail_server/classifier.py`. Given a trace's raw error/route data, classify into one of: `exception` (status="error", error_type not `"retry_loop"`/`"timeout"`), `retry_loop` (error_type == `"retry_loop"`, from F5/F6's `RetryExhaustedError` tagging), `timeout` (error_type == `"timeout"`, from C1), `malformed_output` (status="ok" but `original_route == "unknown"` — F5's `_extract_route` strategy-4 fallback), `none` (a normal resolved-route success). Wire into T2's ingest endpoint: classify on insert, store `failure_category` alongside the record. Extend T1's schema with a migration (`0002_failure_classification.sql`) adding `failure_category` (nullable text) to `traces`, and extend T3's query API to filter by it.
**Files touched:** `src/conntrail_server/classifier.py`, `src/conntrail_server/routes/ingest.py`, `src/conntrail_server/migrations/0002_failure_classification.sql`, `src/conntrail_server/store.py`, `src/conntrail_server/routes/query.py`.
**Acceptance criteria:** Each of the 5 categories above is produced correctly from representative input records. Ingest endpoint persists `failure_category` for every inserted trace (including `"none"` for clean successes). Query API accepts `failure_category` as a filter param.
**Test plan:** `tests/unit/server/test_classifier.py` — one test per category, plus an ambiguous/edge case (e.g. `status="ok"` with a non-"unknown" route is unambiguously `"none"`). `tests/unit/server/test_ingest_route.py` — extend to assert `failure_category` is persisted. No integration test needed (pure function + already-covered ingest path).
**Dependencies:** C1, T1, T2, T3.

---

## Phase 4 — Dashboard

Depends on Phase 2's query API (T3). The before/after panel additionally depends on Phase 5.

### D1 — Trace explorer + per-trace detail + failure view
**Description:** New `src/conntrail_dashboard/` FastAPI + Jinja2 + HTMX app (per §0's stack decision). Pages: trace list (filterable by `node_id`/`stability`/`failure_category`/time range, reading T3's `GET /v1/traces`), per-trace detail (contrasts, entropy, attribution, counterfactual — reading `GET /v1/traces/{id}`), and a failure view (list filtered to `failure_category != "none"`, grouped/counted by category). All server-rendered, HTMX for filter/pagination without full page reloads.
**Files touched:** `src/conntrail_dashboard/app.py`, `src/conntrail_dashboard/routes.py`, `src/conntrail_dashboard/templates/*.html`, `src/conntrail_dashboard/static/*`.
**Acceptance criteria:** Dashboard, pointed at a running collector via `COLLECTOR_URL` env var, renders a live trace list, correctly filters, and shows full per-trace detail including entropy/stability/attribution/counterfactual and (for failed traces) `error_type`/`error_message`/`failure_category`.
**Test plan:** `tests/unit/dashboard/test_routes.py` using FastAPI `TestClient` with a mocked collector HTTP client (list page renders rows, detail page 404s on missing trace, failure view filters correctly). One integration test manually driving a browser is out of scope for this ticket (no browser-automation infra requested) — verify manually against a running collector before marking done, per the general UI-testing guidance.
**Dependencies:** T3, C2 (for failure view's category filter).

### D2 — Before/after CPE-GEPA panel
**Description:** Dashboard page comparing a GEPA run's pre- and post-optimization trace populations (mean entropy, fragile/boundary/confident counts, dominant attribution) — reading from Phase 5's G3 query surface.
**Files touched:** `src/conntrail_dashboard/routes.py` (extend), `src/conntrail_dashboard/templates/before_after.html`.
**Acceptance criteria:** Given a completed GEPA run (G2/G3), the panel renders the first vs. last attempt's aggregate CPE stats side by side, matching the numbers `PromptAttemptRecord`'s `mean_entropy`/`fragile_count`/`boundary_count`/`dominant_attribution` properties compute.
**Test plan:** `tests/unit/dashboard/test_before_after.py` — render against a fixture pair of attempt records (mocked G3 query response), assert computed deltas match expected values. No integration test required beyond D1's manual verification pattern.
**Dependencies:** D1, G3.

---

## Phase 5 — CPE-GEPA (real path)

Unblocked by §0's decision. This is the highest-risk phase: `CPEGEPAOptimizer`/`dspy.GEPA` (`Conntrail-Lib/conntrail/gepa/`) is real code but has **never been run against live data** — the existing test suite mocks `dspy` entirely, and the only live "+10% lift" result on record was produced by a completely different bespoke loop (`run_cpe_gepa.py`), not this optimizer. There is no guarantee this path works against a real `dspy.GEPA` call on first try; budget for debugging, not just running a script.

### G1 — dspy.Module student for one agent
**Description:** `dspy.GEPA` requires a `dspy.Module` student; none of the existing experiment agents (`Conntrail-Lib/experiments/agents/*`) are dspy modules — they're plain LangGraph graphs. Build one new minimal `dspy.Module` wrapping the customer-support routing decision (4-category classifier: refund/escalation/order_info/general — reusing the categories and prompt intent from `Conntrail-Lib/experiments/agents/customer_support/adapter.py` as reference, not imported) plus a `dspy.Example` trainset (reuse `CUSTOMER_SUPPORT_INPUTS`-style examples as reference data, hand-adapted to `dspy.Example` shape). Lives outside the installable packages — this is a runnable example, not library code.
**Files touched:** `examples/gepa/customer_support_student.py`, `examples/gepa/trainset.py`.
**Acceptance criteria:** The `dspy.Module` runs standalone (produces a category prediction for a sample input) before being handed to the optimizer — verifies the module itself is correct independent of GEPA.
**Test plan:** `tests/unit/examples/test_gepa_student.py` — module forward pass produces one of the 4 valid categories for a handful of fixture inputs, using a mocked LM. Not integration (mocked), since this ticket is about the module's structural correctness, not a live run (that's G2).
**Dependencies:** none beyond Phase 1's SDK being available (`conntrail[gepa]` extra).

### G2 — Real `CPEGEPAOptimizer.compile()` run
**Description:** `examples/gepa/run_live.py` — a live run: builds G1's student + trainset, wraps the routing decision with `trace_node()`/`NodeInterceptor` per `CPEGEPAOptimizer`'s documented requirements (`async_mode=False`, `entropy_alert_threshold=0.0`, `sample_rate=1.0` — see `Conntrail-Lib/conntrail/gepa/optimizer.py`'s docstring), and calls `CPEGEPAOptimizer(...).compile()` against a real Anthropic API key. Ports `conntrail/gepa/optimizer.py`, `bridge.py`, `feedback.py`, `schema.py` verbatim into `src/conntrail/gepa/` (they're real and already fully unit-tested with `dspy` mocked — no changes needed, just port + this new live-run consumer).
**Files touched:** `src/conntrail/gepa/optimizer.py`, `bridge.py`, `feedback.py`, `schema.py` (ported), `examples/gepa/run_live.py`.
**Acceptance criteria:** `compile()` completes against live data without raising; `optimizer.attempt_records` contains ≥2 `PromptAttemptRecord`s with real (non-mocked) `TraceRecord`s attached, real `mean_entropy`/`fragile_count` values. **Document the actual measured routing-accuracy delta honestly** — do not carry forward or imply the old repo's "+10%" figure, which came from an unrelated bespoke loop; whatever this run actually measures is the number that goes in any user-facing copy.
**Test plan:** `tests/unit/test_gepa_optimizer.py`, `test_gepa_bridge.py`, `test_gepa_feedback.py` — port the existing fully-mocked unit suites unchanged (still valid, still the right level for this logic). **One integration test**, `tests/integration/test_gepa_live_run.py::test_compile_produces_real_attempts` marked `@pytest.mark.integration` + `@pytest.mark.slow`, running a small (`num_iterations`≈2-3) real `compile()` against G1's student.
**Dependencies:** G1, F6 (interceptor's `on_alert`/sync-mode path), Phase 1 SDK.

### G3 — Persist + expose GEPA run results
**Description:** New collector surface so D2's before/after panel has something to read. `POST /v1/gepa-attempts` (accepts a `PromptAttemptRecord`-shaped payload — attempt_id, prompt_candidate, scalar_score, and the list of `TraceRecord`s, which reuses F4's shape) and `GET /v1/gepa-attempts?run_id=...` for retrieval. New `gepa_attempts` table (migration `0003_gepa_attempts.sql`) referencing `traces` rows by id where possible, or storing trace snapshots inline (simplest: store the full attempt payload as JSON, matching the `traces` table's existing pattern of JSON columns for nested structures).
**Files touched:** `src/conntrail_server/routes/gepa.py`, `src/conntrail_server/migrations/0003_gepa_attempts.sql`, `src/conntrail_server/store.py` (extend).
**Acceptance criteria:** G2's `run_live.py` can POST each completed attempt as it finishes; D2 can GET a run's full attempt history and compute first-vs-last deltas from it.
**Test plan:** `tests/unit/server/test_gepa_route.py` — POST/GET round trip, filter by `run_id`, 404 on unknown run. No integration test required beyond D1/D2's manual-verification pattern (this is straightforward CRUD on an already-proven store pattern from T1).
**Dependencies:** T1, G2.

---

## Phase 6 — Deploy

Last, once Phases 2–4 exist to containerize.

### O1 — `Dockerfile.server`
**Description:** Multi-stage Dockerfile building `conntrail[server]`, running the collector via `uvicorn`. Sqlite file lives on a mounted volume (not baked into the image).
**Files touched:** `deploy/Dockerfile.server`.
**Acceptance criteria:** `docker build` succeeds; container starts and responds to `GET /v1/traces` (with correct auth per T4) with a running volume-mounted sqlite file that persists across container restarts.
**Test plan:** No unit test (infra). Manual verification: build, run, curl the ingest/query endpoints, restart the container, confirm data persisted.
**Dependencies:** T1–T4.

### O2 — `Dockerfile.dashboard`
**Description:** Multi-stage Dockerfile building `conntrail[dashboard]`, running the dashboard app, configured via `COLLECTOR_URL` env var.
**Files touched:** `deploy/Dockerfile.dashboard`.
**Acceptance criteria:** `docker build` succeeds; container starts and serves the trace explorer when pointed at a running collector.
**Test plan:** Manual verification only, same pattern as O1.
**Dependencies:** D1, D2.

### O3 — `docker-compose.yml`
**Description:** Wires collector + dashboard together, sqlite volume, `COLLECTOR_API_KEY`/`COLLECTOR_URL` env wiring, healthchecks for both services.
**Files touched:** `deploy/docker-compose.yml`, `.env.example` (extend with deploy-relevant vars).
**Acceptance criteria:** `docker compose up` brings up both services; dashboard is reachable and shows live data end-to-end from a `trace_node()`-instrumented sample app pointed at the composed collector.
**Test plan:** Manual verification — this is the closest thing to Success Criteria (§8) verification, worth running by hand against a small real traced function before calling the phase done.
**Dependencies:** O1, O2, T1–T4.

---

## Dependency graph (ticket-level)

```
F1 → F2, F3
F2, F3 → F4 → F5 → F6 → F7
F4 → T1 → T2, T3 → T4
T2 → T5 (+ F6)
T1,T2,T3,T4,T5,F6,F7 → T6

F6 → C1 → C2 (also needs T1,T2,T3)

T3, C2 → D1
D1, G3 → D2

(Phase1 SDK) → G1 → G2 (needs F6) → G3 (needs T1)

T1–T4 → O1
D1,D2 → O2
O1,O2,T1–T4 → O3
```

## Open items to confirm before implementation starts

- Dashboard stack (FastAPI+Jinja2+HTMX vs. a JS SPA) — my default, not an answered open question. Say the word if you'd rather go SPA.
- G1's choice of customer_support as the pilot agent — swap for a different domain if you have a preference (financial_query and medical_triage are the other two with real experiment data in `Conntrail-Lib`).
- Whether `examples/gepa/` belongs in this repo at all vs. a separate demo repo — kept in-repo here since Phase 5 needs somewhere to live and there's no SequenceBench-style demo target yet (§9).
