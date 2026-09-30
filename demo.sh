#!/usr/bin/env bash
#
# Conntrail demo runner.
#
# Brings up the collector + dashboard, traces a real external agent (the
# AI-Agents-Sequence-Game-Tournament, used read-only), and runs the
# cost-weighted GEPA optimization with a weak seed so the before/after panel
# has a visible improvement to show.
#
# Usage:
#   ./demo.sh services   start collector + dashboard in the background
#   ./demo.sh stop       stop them
#   ./demo.sh reset      stop them and delete the demo database
#   ./demo.sh trace      run the Sequence-agent tracing pass
#   ./demo.sh gepa       run the weak-seed GEPA cost comparison (lambda 0 vs 0.1)
#   ./demo.sh all        services + trace + gepa
#   ./demo.sh urls       print the dashboard URLs
#
# Provider selection (first match wins):
#   CONNTRAIL_SEQUENCE_MODEL / CONNTRAIL_GEPA_* env overrides
#   else OPENROUTER_API_KEY set  -> OpenRouter models (real token + $ telemetry)
#   else LOCAL_LLM_URL set       -> local/<LOCAL_MODEL_NAME> via the local server
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DEMO_DIR="$ROOT/.demo"
LOG_DIR="$DEMO_DIR/logs"
mkdir -p "$LOG_DIR"

# Caller-supplied CONNTRAIL_* overrides win over .env, so a demo can run
# against a different provider without editing .env (e.g.
# `CONNTRAIL_SEQUENCE_MODEL=openrouter/... ./demo.sh all`).
_OVERRIDE_VARS=(
  CONNTRAIL_SEQUENCE_MODEL
  CONNTRAIL_CONTRAST_MODEL
  CONNTRAIL_GEPA_STUDENT_MODEL
  CONNTRAIL_GEPA_REFLECTION_MODEL
)
declare -A _OVERRIDES=()
for _v in "${_OVERRIDE_VARS[@]}"; do
  if [[ -n "${!_v:-}" ]]; then _OVERRIDES[$_v]="${!_v}"; fi
done

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

for _v in "${!_OVERRIDES[@]}"; do export "$_v=${_OVERRIDES[$_v]}"; done

PY="python3"
if [[ -x .venv/bin/python ]]; then PY=".venv/bin/python"; fi

COLLECTOR_PORT="${COLLECTOR_PORT:-8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8001}"
COLLECTOR_URL="${COLLECTOR_URL:-http://127.0.0.1:${COLLECTOR_PORT}}"
DASHBOARD_URL="${DASHBOARD_URL:-http://127.0.0.1:${DASHBOARD_PORT}}"
export COLLECTOR_DB_PATH="${COLLECTOR_DB_PATH:-$DEMO_DIR/conntrail.sqlite3}"
export COLLECTOR_URL DASHBOARD_URL

wait_for() {
  local url="$1" name="$2"
  for _ in $(seq 1 40); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "  $name ready"
      return 0
    fi
    sleep 0.5
  done
  echo "ERROR: $name did not become ready — check $LOG_DIR" >&2
  return 1
}

services() {
  if curl -fsS "$COLLECTOR_URL/healthz" >/dev/null 2>&1; then
    echo "  collector already up at $COLLECTOR_URL"
  else
    echo "  starting collector on port $COLLECTOR_PORT (db: $COLLECTOR_DB_PATH)"
    nohup "$PY" -m uvicorn conntrail_server.app:create_app --factory --port "$COLLECTOR_PORT" \
      >"$LOG_DIR/collector.log" 2>&1 &
    echo $! >"$DEMO_DIR/collector.pid"
  fi
  if curl -fsS "$DASHBOARD_URL/" >/dev/null 2>&1; then
    echo "  dashboard already up at $DASHBOARD_URL"
  else
    echo "  starting dashboard on port $DASHBOARD_PORT"
    nohup "$PY" -m uvicorn conntrail_dashboard.app:create_app --factory --port "$DASHBOARD_PORT" \
      >"$LOG_DIR/dashboard.log" 2>&1 &
    echo $! >"$DEMO_DIR/dashboard.pid"
  fi
  wait_for "$COLLECTOR_URL/healthz" "collector"
  wait_for "$DASHBOARD_URL/" "dashboard"
}

stop() {
  for svc in collector dashboard; do
    local pidfile="$DEMO_DIR/$svc.pid"
    if [[ -f "$pidfile" ]]; then
      local pid
      pid="$(cat "$pidfile")"
      if kill "$pid" 2>/dev/null; then
        echo "  stopping $svc (pid $pid)"
        # Wait for it to actually exit — otherwise a following `services`
        # can see the dying process answer /healthz and skip restarting.
        for _ in $(seq 1 40); do
          if ! kill -0 "$pid" 2>/dev/null; then break; fi
          sleep 0.25
        done
        if kill -0 "$pid" 2>/dev/null; then
          echo "  $svc did not exit — sending SIGKILL"
          kill -9 "$pid" 2>/dev/null || true
        fi
      fi
      rm -f "$pidfile"
    fi
  done
}

urls() {
  echo
  echo "  collector : $COLLECTOR_URL/healthz"
  echo "  dashboard : $DASHBOARD_URL/            (Trace Explorer)"
  echo "              $DASHBOARD_URL/cost          (per-node cost view)"
  echo "              $DASHBOARD_URL/before-after  (GEPA before/after)"
  echo
}

pick_models() {
  if [[ -n "${CONNTRAIL_SEQUENCE_MODEL:-}" ]]; then
    SEQUENCE_MODEL="$CONNTRAIL_SEQUENCE_MODEL"
  elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    SEQUENCE_MODEL="openrouter/deepseek/deepseek-v4.1-flash"
  elif [[ -n "${LOCAL_LLM_URL:-}" ]]; then
    SEQUENCE_MODEL="local/${LOCAL_MODEL_NAME:-local-model}"
  else
    echo "ERROR: set OPENROUTER_API_KEY (or LOCAL_LLM_URL), or CONNTRAIL_SEQUENCE_MODEL." >&2
    exit 1
  fi
  CONTRAST_MODEL="${CONNTRAIL_CONTRAST_MODEL:-$SEQUENCE_MODEL}"

  if [[ -n "${CONNTRAIL_GEPA_STUDENT_MODEL:-}" ]]; then
    STUDENT_MODEL="$CONNTRAIL_GEPA_STUDENT_MODEL"
  elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    STUDENT_MODEL="openrouter/deepseek/deepseek-v4.1-flash"
  else
    STUDENT_MODEL="local/${LOCAL_MODEL_NAME:-local-model}"
  fi

  if [[ -n "${CONNTRAIL_GEPA_REFLECTION_MODEL:-}" ]]; then
    REFLECTION_MODEL="$CONNTRAIL_GEPA_REFLECTION_MODEL"
  elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    REFLECTION_MODEL="openrouter/deepseek/deepseek-v4.1-flash"
  else
    REFLECTION_MODEL="local/${LOCAL_MODEL_NAME:-local-model}"
  fi
  export CONNTRAIL_SEQUENCE_MODEL="$SEQUENCE_MODEL"
  export CONNTRAIL_CONTRAST_MODEL="$CONTRAST_MODEL"
}

maybe_load_local_model() {
  # Only when a local/* model is actually selected — LOCAL_AUTH_MODE=jwt can
  # be configured in .env for a local rig while a demo runs against OpenRouter.
  if [[ "${SEQUENCE_MODEL:-}" == local/* && "${LOCAL_AUTH_MODE:-}" == "jwt" && -n "${LOCAL_MODEL_NAME:-}" ]]; then
    echo "  ensuring local model '${LOCAL_MODEL_NAME}' is loaded ..."
    "$PY" - <<'PYEOF'
import os

import httpx

from conntrail.utils.providers import _get_local_token, _LOCAL_LLM_URL

base = _LOCAL_LLM_URL.rstrip("/v1").rstrip("/")
token = _get_local_token()
response = httpx.post(
    f"{base}/v1/load",
    json={"model_path": os.environ["LOCAL_MODEL_NAME"]},
    headers={"Authorization": f"Bearer {token}"},
    timeout=300,
)
print(f"    load: {response.status_code} {response.text[:100]}")
PYEOF
  fi
}

api_key_arg() {
  if [[ -n "${COLLECTOR_API_KEY:-}" ]]; then
    printf -- '--collector-api-key\n%s\n' "$COLLECTOR_API_KEY"
  fi
}

trace() {
  pick_models
  maybe_load_local_model
  echo
  echo "== tracing the Sequence-game agent (model: $SEQUENCE_MODEL) =="
  local extra=()
  if [[ "$SEQUENCE_MODEL" == local/* ]]; then
    extra+=(--skip-native)
    echo "   (local model: skipping the slow native flavor)"
  fi
  # shellcheck disable=SC2046
  "$PY" examples/sequence/trace_sequence_agent.py \
    --model "$SEQUENCE_MODEL" \
    --contrast-model "$CONTRAST_MODEL" \
    --collector-url "$COLLECTOR_URL" \
    $(api_key_arg) \
    "${extra[@]}"
}

gepa() {
  pick_models
  maybe_load_local_model
  echo
  echo "== weak-seed GEPA cost comparison (student: $STUDENT_MODEL, reflection: $REFLECTION_MODEL) =="
  local common=(--task policy --weak-seed --num-examples 30 --max-metric-calls 24 --sample-rate 1.0 --holdout 8 --collector-url "$COLLECTOR_URL")
  # shellcheck disable=SC2207
  common+=($(api_key_arg))
  for w in 0 0.1; do
    local log="$LOG_DIR/gepa_cw${w}.log" out="$DEMO_DIR/gepa_cw${w}.json"
    echo "-- cost-weight $w -> $log"
    "$PY" examples/gepa/run_live.py "${common[@]}" \
      --student-model "$STUDENT_MODEL" \
      --reflection-model "$REFLECTION_MODEL" \
      --contrast-model "$CONTRAST_MODEL" \
      --cost-weight "$w" \
      --output "$out" 2>&1 | tee "$log"
    local run_id
    run_id="$(grep -oE 'run_id=[a-f0-9-]+' "$log" | head -1 | cut -d= -f2 || true)"
    if [[ -n "$run_id" ]]; then
      echo "   before/after: $DASHBOARD_URL/before-after?run_id=$run_id"
    fi
  done
}

case "${1:-all}" in
  services)
    services
    urls
    ;;
  stop)
    stop
    ;;
  reset)
    stop
    echo "  removing $COLLECTOR_DB_PATH"
    rm -f "$COLLECTOR_DB_PATH"
    ;;
  trace)
    services
    trace
    urls
    ;;
  gepa)
    services
    gepa
    urls
    ;;
  all)
    services
    trace
    gepa
    urls
    ;;
  urls)
    urls
    ;;
  *)
    echo "usage: $0 {services|stop|reset|trace|gepa|all|urls}" >&2
    exit 1
    ;;
esac
