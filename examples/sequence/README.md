# Sequence-game tracing harness

Instruments the strategic routing decision of the
[AI-Agents-Sequence-Game-Tournament](../../../AI-Agents-Sequence-Game-Tournament)
agent with Conntrail — **read-only**: nothing in the tournament project is
modified.

The tournament's full board prompt is too large for contrast generation, so
this harness traces the agent's *decision*, not its full move generation:

1. A real game position is replayed through the game engine (seeded
   random-vs-random) and distilled — with the game's own `Game/sequence/analysis.py`
   functions — into a compact situation summary.
2. The agent's LLM, given the project's verbatim strategy system prompt
   (`AI/prompt.py::SYSTEM_PROMPT`), answers which priority it follows:
   `WIN / BLOCK / FORK / ADVANCE / CENTER / DISRUPT`.
3. That decision node is wrapped in Conntrail's `trace_node()`.

Each run produces three traced node flavors per position:

| Node | What it shows |
|---|---|
| `sequence_strategy_native` | The project's own provider plumbing (raw openai SDK). Non-LangChain, so cost capture is **best-effort by design** — latency and observer overhead recorded, node-internal tokens not visible. |
| `sequence_strategy_langchain` | The same verbatim prompt through Conntrail's provider layer — **full token / cache-read / cost capture**. |
| `sequence_strategy_error` | A deliberate mid-decision failure; records the error trace (usage captured before the raise) and re-raises. |

## Prerequisites

- A running collector + dashboard (`./demo.sh services`, or docker compose).
- An LLM: either `OPENROUTER_API_KEY` (recommended — real token *and dollar*
  telemetry) or a local OpenAI-compatible server (`LOCAL_LLM_URL`, e.g.
  Unsloth Studio / Ollama / llama.cpp).
- The tournament repo checked out as a sibling of this repo (or pass
  `--game-repo` / set `SEQUENCE_GAME_REPO`).

## Usage

```bash
# OpenRouter
export OPENROUTER_API_KEY=...
python examples/sequence/trace_sequence_agent.py \
    --model openrouter/deepseek/deepseek-v4.1-flash

# Local server (native flavor is slow on local thinking models — skip it)
python examples/sequence/trace_sequence_agent.py \
    --model local/unsloth/gemma-4-12b-it-GGUF --skip-native
```

Useful flags: `--contrast-model`, `--collector-url` / `--collector-api-key`,
`--depths 8,14,22`, `--game-repo`, `--no-verify`.

After the run the script queries the collector and prints every trace's
stability, token usage, cache reads, estimated cost, observer overhead, and
cost findings — then the per-node cost summary and any cross-node shared
instruction blocks it found.

`./demo.sh` runs this plus the GEPA comparison and starts the services.
