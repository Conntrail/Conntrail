# AGENTS.md

Guidance for working in the Conntrail repo. Read `README.md` and `docs/TESTING.md` for full context; this file captures the non-obvious, easy-to-miss facts.

## Layout (one repo, four installable packages)

- `src/conntrail/` — SDK: `trace_node()`/`trace_graph()` in `wrap.py`, `interceptor.py`, `contrast.py`, `analyser.py`, `record.py`, `config.py`, `cost.py` (token usage capture, price table, LangChain cost callback handler), `cost_analyzer.py` (derived cost findings), `utils/providers.py`, `utils/entropy.py`, `gepa/`. Public API is only `trace_node`, `trace_graph`, `ConntrailConfig` (re-exported in `src/conntrail/__init__.py`).
- `src/conntrail_server/` — collector: FastAPI app (`app.py`), `routes/` (ingest/query/gepa/cost), `store.py`, `classifier.py`, `auth.py`, `migrations/`.
- `src/conntrail_dashboard/` — dashboard: FastAPI + Jinja2 + HTMX (`app.py`, `routes/`, `client.py`, `templates/`, `static/`). No JS build step.
- `examples/gepa/`, `deploy/` (Dockerfiles + compose), `docs/` (EPICS.md work breakdown, TESTING.md test plan + verified results).

## Run / test commands

```bash
pytest tests/unit -m "not integration"        # pure unit, no LLM (~3 s)
pytest tests/unit/test_contrast.py -m integration   # live contrast only
pytest tests/integration -m "integration or slow"   # live-LLM round trips
ruff check .
```

- `.env` is loaded automatically by `tests/conftest.py` (`load_dotenv`), so live tests pick up API keys without export. Never commit real keys — `.env` is gitignored.
- Markers `integration` and `slow` are registered in both `pyproject.toml` and `conftest.py`.
- `asyncio_mode = "auto"`, `pythonpath = ["src", "."]` — no manual `sys.path` or `@pytest.mark.asyncio` needed.
- Use `.venv/bin/python` / `.venv/bin/pytest` (Python 3.14).

## Serve the apps

Both use `--factory` (the app is built by a `create_app()` factory, not a module-level object):

```bash
uvicorn conntrail_server.app:create_app --factory --port 8000
uvicorn conntrail_dashboard.app:create_app --factory --port 8001
```

## Gotchas

- **Migrations are plain `.sql` files** in `src/conntrail_server/migrations/`, applied in filename order and tracked in a `schema_migrations` table (`migrations/__init__.py::run_migrations`). No ORM/Alembic. Add a new numbered file for schema changes, never edit applied ones.
- **LLM provider resolution** goes through `conntrail.utils.providers.get_chat_model()`. Provider is inferred from the model-name prefix (`claude-*`, `gpt-*`, `gemini-*`, `llama-*`, `openrouter/*`, `local/*`). Cloud key priority: GROQ → ANTHROPIC → OPENAI → GOOGLE → OPENROUTER. `local/<name>` targets an OpenAI-compatible local server (`LOCAL_LLM_URL`, `LOCAL_AUTH_MODE=none|api_key|jwt`). Token-usage streaming is per-provider: all ChatOpenAI-based clients pass `stream_usage=True` explicitly (langchain-openai auto-enables it ONLY for the default api.openai.com base URL — custom local/OpenRouter base URLs silently drop usage without it); Anthropic/Groq/Gemini report usage natively.
- **Default contrast model** is `claude-haiku-4-5-20251001` (defaulted in `config.py:37` and `contrast.py:57`). Live tests override via `CONNTRAIL_CONTRAST_MODEL`.
- **Integration tests skip** unless a cloud key **or** an explicit `LOCAL_LLM_URL` is set (`conftest.py::live_llm_available`). An implicit provider default URL doesn't count.
- **Local Unsloth/JWT caveats** (`docs/TESTING.md` §2): model must be loaded on the server; JWT mode disables chain-of-thought because reasoning models burn the whole `max_tokens` budget; dspy holds one token per run and can outlive the TTL (LangChain path re-auths per call).
- **Failure classification** (`conntrail_server/classifier.py`) is derived from SDK signals, not invented: `exception`, `retry_loop`, `timeout`, `malformed_output`, `none`. Don't invent categories. Same rule for cost findings (`conntrail/cost_analyzer.py`): only `cache_efficiency`, `prompt_size`, `repeated_instructions`, `output_discipline`, `observer_overhead`, severities `info`/`warning` — findings recommend, never auto-apply changes.
- **Cost capture is LangChain best-effort** (`conntrail/cost.py`): a callback handler injected via the `var_child_runnable_config` contextvar sees only LLM calls made through LangChain clients with no explicit config. dspy students are invisible to it — `examples/gepa/run_live.py` stamps attempt-level usage with dspy's own `track_usage()` instead (covers hot path + the 4 analysis re-runs; the constant cancels in baseline-relative scoring).
- **The price table** (`cost.py::_MODEL_PRICES`) is approximate $/Mtok by longest-prefix model match; `local/*` is free. Corrections go via `ConntrailConfig.model_prices` or the `CONNTRAIL_PRICE_OVERRIDES` env var (JSON) — never treat `cost_usd` as an invoice.
- **GEPA cost scoring**: `score = task − cost_weight × (tokens/baseline − 1)` (baseline = first attempt) in `feedback.py`, plus `objective_scores={"cost": −tokens}` for gepa's Pareto tracking. `PromptAttemptRecord.scalar_score` always stays the RAW task score — only the returned Prediction's score is cost-adjusted.

## Style / config

- `pyproject.toml`: ruff, `line-length = 100`, `select = ["E","F","I","UP"]`, `ignore = ["E501"]`. `ruff check .` must stay clean.
- Lint clean + unit suite passing is the standard before-commit check. Live/integration tests are optional gated by keys.
