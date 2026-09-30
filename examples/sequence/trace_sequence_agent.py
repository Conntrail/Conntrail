"""
Sequence-game tracing harness — instruments the strategic routing decision of
the AI-Agents-Sequence-Game-Tournament agent with Conntrail.

Read-only: nothing in the tournament project is modified. Following
docs/TESTING.md §5, the tournament's full board prompt is too large for
contrast generation, so this harness traces the agent's *strategic routing
decision*:

  1. A real game position is replayed through the game engine (seeded
     random-vs-random) and distilled — with the game's own analysis
     functions — into a compact situation summary.
  2. The agent's LLM, using the project's verbatim strategy system prompt
     (AI/prompt.py::SYSTEM_PROMPT), answers which priority it follows:
     WIN / BLOCK / FORK / ADVANCE / CENTER / DISRUPT.
  3. That decision node is wrapped in Conntrail's trace_node(), so each
     position produces a TraceRecord with stability analysis *and* the new
     cost telemetry (tokens, cache reads, findings, observer overhead).

Three node flavors run per position:

  sequence_strategy_native
      The project's own provider plumbing (raw openai SDK) against the same
      endpoint — exactly the tournament's call path. Non-LangChain, so
      Conntrail's cost capture is best-effort by design: latency and the
      observer's own overhead are recorded, but the node's internal token
      usage is not visible.
  sequence_strategy_langchain
      The same verbatim prompt routed through Conntrail's provider layer
      (get_chat_model) — full token/cache/cost capture.
  sequence_strategy_error
      A deliberate mid-decision failure: the LLM answers, then the node
      raises. Records an error trace (usage is captured before the failure)
      and re-raises the original exception.

Usage:
    # OpenRouter — real token and dollar telemetry:
    export OPENROUTER_API_KEY=...
    python examples/sequence/trace_sequence_agent.py \
        --model openrouter/deepseek/deepseek-v4.1-flash

    # Local OpenAI-compatible server (Unsloth / Ollama / llama.cpp / vLLM):
    python examples/sequence/trace_sequence_agent.py \
        --model local/unsloth/gemma-4-12b-it-GGUF --skip-native

The tournament repo is located via --game-repo, the SEQUENCE_GAME_REPO env
var, or autodetection among a few sibling paths. A running collector and
dashboard make the result visible (`demo.sh` orchestrates the whole flow).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from conntrail import ConntrailConfig, trace_node  # noqa: E402
from conntrail.exporters.http import HttpExporter  # noqa: E402
from conntrail.utils.providers import _resolve_local_api_key, get_chat_model  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

DEFAULT_MODEL = "local/unsloth/gemma-4-12b-it-GGUF"
PRIORITIES = ("WIN", "BLOCK", "FORK", "ADVANCE", "CENTER", "DISRUPT")
DEFAULT_DEPTHS = (8, 14, 22)

# The verbatim strategy system prompt demands "respond with ONLY a JSON
# object", so the priority question is asked in JSON form too — answering
# with a bare word gets overridden by the system prompt's formatting rule.
PRIORITY_QUESTION = (
    "Which strategic priority do you follow right now? Respond with ONLY this "
    'JSON object: {"priority": "<your choice>"} — where <your choice> is exactly '
    "one word from: WIN, BLOCK, FORK, ADVANCE, CENTER, DISRUPT."
)

_GAME_REPO_CANDIDATES = (
    REPO_ROOT.parent / "AI-Agents-Sequence-Game-Tournament",
    REPO_ROOT.parent.parent / "AI-Agents-Sequence-Game-Tournament",
    Path.home() / "AI-Agents-Sequence-Game-Tournament",
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without the tournament repo)
# ---------------------------------------------------------------------------


def parse_priority(text: str) -> str:
    """Extract the chosen priority from an LLM response (first label wins).

    Tolerates inline reasoning and JSON wrappers. Returns "unknown" when no
    label is present — which the cost analyzer treats as a malformed-output
    signal, exactly the behavior a real agent would exhibit.
    """
    cleaned = re.sub(r"</?think>.*?</?think>", "", text or "", flags=re.DOTALL).upper()
    best_pos, best_label = len(cleaned), "unknown"
    for label in PRIORITIES:
        pos = cleaned.find(label)
        if pos != -1 and pos < best_pos:
            best_pos, best_label = pos, label
    return best_label


def resolve_game_repo(cli_arg: str | None) -> Path:
    """Locate the tournament repo: --game-repo > SEQUENCE_GAME_REPO > autodetect."""
    if cli_arg:
        repo = Path(cli_arg).expanduser()
        if not (repo / "AI" / "prompt.py").is_file():
            raise SystemExit(f"--game-repo {repo} does not look like the tournament repo")
        return repo
    env = os.environ.get("SEQUENCE_GAME_REPO")
    if env:
        return resolve_game_repo(env)
    for candidate in _GAME_REPO_CANDIDATES:
        if (candidate / "AI" / "prompt.py").is_file():
            return candidate
    raise SystemExit(
        "Could not locate the AI-Agents-Sequence-Game-Tournament repo. Pass "
        "--game-repo /path/to/repo or set SEQUENCE_GAME_REPO."
    )


def native_target(model: str) -> tuple[str, str, str]:
    """(base_url, api_key, api_model) for the tournament's own OpenAIProvider.

    Only OpenAI-compatible endpoints can use the project's native plumbing:
    local servers ("local/<name>") and OpenRouter ("openrouter/<vendor>/<model>").
    """
    if model == "local" or model.startswith("local/"):
        name = model.split("/", 1)[1] if "/" in model else os.environ.get(
            "LOCAL_MODEL_NAME", "local-model"
        )
        return (
            os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8888/v1"),
            _resolve_local_api_key(),
            name,
        )
    if model.startswith("openrouter/"):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is required for an openrouter/* model.")
        return ("https://openrouter.ai/api/v1", key, model.split("/", 1)[1])
    raise SystemExit(
        f"model {model!r} has no OpenAI-compatible endpoint for the native flavor; "
        "use --skip-native (the langchain flavor supports any Conntrail model)."
    )


def load_game_modules(game_repo: Path) -> SimpleNamespace:
    """Import the tournament project's modules after validating the repo path."""
    if str(game_repo) not in sys.path:
        sys.path.insert(0, str(game_repo))
    from AI.prompt import SYSTEM_PROMPT
    from AI.providers import get_provider
    from Game.sequence.analysis import (
        find_completing_positions,
        find_fork_positions,
        find_partial_sequences,
        find_threat_positions,
        get_key_center_positions,
    )
    from Game.sequence.game import apply_move, get_moves_for_current_player
    from Game.sequence.random_agent import random_agent
    from Game.sequence.state import init_game

    return SimpleNamespace(
        SYSTEM_PROMPT=SYSTEM_PROMPT,
        get_provider=get_provider,
        find_completing_positions=find_completing_positions,
        find_fork_positions=find_fork_positions,
        find_partial_sequences=find_partial_sequences,
        find_threat_positions=find_threat_positions,
        get_key_center_positions=get_key_center_positions,
        apply_move=apply_move,
        get_moves_for_current_player=get_moves_for_current_player,
        random_agent=random_agent,
        init_game=init_game,
    )


# ---------------------------------------------------------------------------
# Real positions + situation distillation (the game's own analysis functions)
# ---------------------------------------------------------------------------


def make_position(game: SimpleNamespace, seed: int, target_turn: int) -> Any:
    """Replay a seeded random-vs-random game to a genuine mid-game position."""
    random.seed(seed)
    state = game.init_game(["p1", "p2"], {"p1": "blue", "p2": "green"})
    while not state.winner and state.turn_number < target_turn:
        moves = game.get_moves_for_current_player(state)
        if not moves:
            break
        state = game.apply_move(state, game.random_agent(moves))  # returns a new state
    return state


def distill_situation(game: SimpleNamespace, state: Any) -> str:
    """Compact situation summary built from the game's own analysis functions."""
    board = state.board
    me_pid = state.current_player
    opp_pid = next(p for p in state.chips if p != me_pid)
    me, opp = state.chips[me_pid], state.chips[opp_pid]

    my_wins = sorted(game.find_completing_positions(board, me))
    threats = sorted(game.find_threat_positions(board, opp))
    forks = sorted(game.find_fork_positions(board, me))
    mine_ps = game.find_partial_sequences(board, me)
    theirs_ps = game.find_partial_sequences(board, opp)
    center_open = sorted(p for p in game.get_key_center_positions() if board.get(p) is None)[:6]

    def fmt_ps(items: list) -> str:
        if not items:
            return "none"
        return "; ".join(
            f"run of {s['length']} at {s['positions']}, extendable at {s['extending_positions']}"
            for s in items[:3]
        )

    lines = [
        f"Turn {state.turn_number}. You have {state.sequences.get(me_pid, 0)} completed "
        f"sequence(s); opponent has {state.sequences.get(opp_pid, 0)}. First to 2 sequences wins.",
        f"Your WIN positions (complete your next sequence): {my_wins or 'none'}.",
        f"Opponent threat positions (they would complete a sequence there): {threats or 'none'}.",
        f"Your FORK positions (complete two sequences at once): {forks or 'none'}.",
        f"Your near-sequences: {fmt_ps(mine_ps)}",
        f"Opponent near-sequences: {fmt_ps(theirs_ps)}",
        f"Open center positions (rows/cols 3-6): {center_open or 'none'}.",
        "",
        PRIORITY_QUESTION,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Traced nodes
# ---------------------------------------------------------------------------


def make_native_node(game: SimpleNamespace, model: str, base_url: str, api_key: str, api_model: str):
    """The project's own provider plumbing (raw openai SDK) — non-LangChain."""

    def sequence_strategy_native(state: dict) -> dict:
        provider = game.get_provider("openai", api_key=api_key, base_url=base_url)
        result = provider.complete(
            messages=[
                {"role": "system", "content": game.SYSTEM_PROMPT},
                {"role": "user", "content": state["message"]},
            ],
            model=api_model,
            temperature=0.3,
            max_tokens=512,
        )
        return {**state, "route": parse_priority(result.content)}

    return sequence_strategy_native


def make_langchain_node(game: SimpleNamespace, model: str):
    """The same verbatim prompt through Conntrail's own provider layer."""

    async def sequence_strategy_langchain(state: dict) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_chat_model(model, max_tokens=200)
        message = await llm.ainvoke(
            [SystemMessage(content=game.SYSTEM_PROMPT), HumanMessage(content=state["message"])]
        )
        return {**state, "route": parse_priority(message.content)}

    return sequence_strategy_langchain


def make_error_node(game: SimpleNamespace, model: str):
    """Deliberate mid-decision failure — usage is captured before the raise."""

    async def sequence_strategy_error(state: dict) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_chat_model(model, max_tokens=200)
        message = await llm.ainvoke(
            [SystemMessage(content=game.SYSTEM_PROMPT), HumanMessage(content=state["message"])]
        )
        route = parse_priority(message.content)
        raise ValueError(f"simulated mid-decision failure after resolving route={route!r}")

    return sequence_strategy_error


# ---------------------------------------------------------------------------
# Wiring + run + verify
# ---------------------------------------------------------------------------


def make_config(args: argparse.Namespace) -> ConntrailConfig:
    exporter = None
    if args.collector_url:
        exporter = HttpExporter(args.collector_url, api_key=args.collector_api_key)
    return ConntrailConfig(
        contrast_model=args.contrast_model or args.model,
        exporter=exporter,
        sample_rate=1.0,
        async_mode=False,
        entropy_alert_threshold=0.0,
    )


async def run(args: argparse.Namespace) -> None:
    game = load_game_modules(resolve_game_repo(args.game_repo))
    config = make_config(args)

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    positions = [make_position(game, seed=(i + 1) * 11, target_turn=d) for i, d in enumerate(depths)]
    situations = [distill_situation(game, p) for p in positions]

    nodes: list[tuple[str, Any]] = []
    if not args.skip_native:
        base_url, api_key, api_model = native_target(args.model)
        nodes.append(
            ("native", trace_node(config=config, input_key="message", route_key="route")(
                make_native_node(game, args.model, base_url, api_key, api_model)
            ))
        )
    nodes.append(
        ("langchain", trace_node(config=config, input_key="message", route_key="route")(
            make_langchain_node(game, args.model)
        ))
    )
    error_node = trace_node(config=config, input_key="message", route_key="route")(
        make_error_node(game, args.model)
    )

    for i, (position, situation) in enumerate(zip(positions, situations), 1):
        print(f"\n=== position {i} (turn {position.turn_number}) ===", flush=True)
        print(f"    {situation.splitlines()[0]}", flush=True)
        for flavor, traced in nodes:
            t0 = time.time()
            try:
                out = await traced({"message": situation, "route": None})
                print(f"  [{flavor:>9}] route={out['route']:<8} ({time.time() - t0:.1f}s)", flush=True)
            except Exception as exc:  # noqa: BLE001 - harness robustness
                print(f"  [{flavor:>9}] FAILED ({time.time() - t0:.1f}s): {exc}", flush=True)

    print("\n=== error path (raises mid-decision) ===", flush=True)
    try:
        await error_node({"message": situations[0], "route": None})
    except ValueError as exc:
        print(f"  re-raised to caller as expected: {exc}", flush=True)


def verify(args: argparse.Namespace) -> None:
    if not args.collector_url:
        print("\n(no collector configured — skipping verification)")
        return
    import httpx

    headers = {"X-API-Key": args.collector_api_key} if args.collector_api_key else {}
    base = args.collector_url.rstrip("/")

    try:
        rows = httpx.get(
            f"{base}/v1/traces", params={"limit": 50}, headers=headers, timeout=15
        ).json()["traces"]
    except httpx.HTTPError as exc:
        print(f"\n(collector unreachable at {base} — skipping verification: {exc})")
        return
    seq_rows = [t for t in rows if t["node_id"].startswith("sequence_strategy")]
    print(f"\n\n========== collector: {len(seq_rows)} sequence-harness traces ==========", flush=True)
    for t in seq_rows:
        usage = t.get("token_usage") or {}
        overhead = t.get("analysis_overhead") or {}
        findings = t.get("cost_findings") or []
        print(
            f"\n{t['node_id']}  [{t['status']}]  route={t['original_route']}  "
            f"entropy={t['entropy_score']:.2f} ({t['stability']})",
            flush=True,
        )
        if usage:
            print(
                f"  node usage : {usage.get('input_tokens')} in / {usage.get('output_tokens')} out "
                f"| {usage.get('cached_input_tokens')} cache-read | {usage.get('llm_call_count')} "
                f"call(s) | models={usage.get('models')}",
                flush=True,
            )
        else:
            print("  node usage : <none — non-LangChain node, best-effort by design>", flush=True)
        cost = t.get("cost_usd")
        print(
            f"  node cost  : ${cost:.6f}" if cost is not None else "  node cost  : <unknown>",
            flush=True,
        )
        print(f"  latency    : {t.get('latency_ms')} ms", flush=True)
        if overhead:
            print(
                f"  overhead   : {overhead.get('total_tokens')} tokens "
                f"(retries={overhead.get('retries')}, {overhead.get('latency_ms')} ms)",
                flush=True,
            )
        for f in findings:
            print(f"  [{f['severity']:7}] {f['dimension']}: {f['evidence'][:100]}", flush=True)

    try:
        summary = httpx.get(f"{base}/v1/cost-summary", headers=headers, timeout=15).json()
    except httpx.HTTPError as exc:
        print(f"\n(collector unreachable for cost summary: {exc})")
        return
    print(f"\n========== cost summary (scanned {summary['scanned_traces']} traces) ==========", flush=True)
    for n in summary["nodes"]:
        if not n["node_id"].startswith("sequence_strategy"):
            continue
        print(
            f"  {n['node_id']}: traces={n['trace_count']} calls={n['llm_call_count']} "
            f"in={n['input_tokens']} cache-read={n['cached_input_tokens']} out={n['output_tokens']} "
            f"hit={n['cache_hit_ratio']} cost=${n['total_cost_usd'] or 0:.6f} "
            f"latency={n['mean_latency_ms']}ms warnings={n['cost_warning_count']}",
            flush=True,
        )
    shared = [b for b in summary["shared_prompt_blocks"] if "sequence_strategy" in " ".join(b["node_ids"])]
    for b in shared:
        print(f"  shared instruction block {b['hash']}: nodes={b['node_ids']} x{b['occurrences']}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--game-repo",
        default=os.environ.get("SEQUENCE_GAME_REPO"),
        help="Path to AI-Agents-Sequence-Game-Tournament (default: SEQUENCE_GAME_REPO env, then autodetect).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CONNTRAIL_SEQUENCE_MODEL", DEFAULT_MODEL),
        help="Traced model: 'local/<name>' or 'openrouter/<vendor>/<model>' "
        "(default: CONNTRAIL_SEQUENCE_MODEL, then the local Unsloth default).",
    )
    parser.add_argument(
        "--contrast-model",
        default=os.environ.get("CONNTRAIL_CONTRAST_MODEL"),
        help="Contrast-generation model (default: --model).",
    )
    parser.add_argument("--collector-url", default=os.environ.get("COLLECTOR_URL"))
    parser.add_argument("--collector-api-key", default=os.environ.get("COLLECTOR_API_KEY"))
    parser.add_argument(
        "--depths",
        default=",".join(str(d) for d in DEFAULT_DEPTHS),
        help="Comma-separated turn depths to replay positions at (default: 8,14,22).",
    )
    parser.add_argument(
        "--skip-native",
        action="store_true",
        help="Skip the project-native flavor (its raw-SDK calls are visible only as "
        "latency; also much slower on local thinking models).",
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip the post-run collector verification."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))
    if not args.no_verify:
        verify(args)


if __name__ == "__main__":
    main()
