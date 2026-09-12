"""
Cost telemetry — token usage capture, pricing, and cost estimation.

Conntrail measures what a traced node *costs*, not just how stable it is:

  - TokenUsage: normalized input/output/cached/cache-write token counts,
    extracted from LangChain AIMessage objects (``usage_metadata`` first,
    then provider ``response_metadata`` shapes).
  - CostCallbackHandler: a LangChain callback handler injected around node
    calls (via the child-runnable config contextvar) that observes every
    LLM call the node makes internally — tokens, cache hits, prompt sizes —
    without touching the node's code. Best-effort: non-LangChain nodes
    simply report no usage.
  - MODEL_PRICES: an approximate USD-per-1M-token price table
    (input, cached-input, output) resolved by longest-prefix match.
    ``local/*`` models are free; unknown models fall back to a conservative
    default. Override entries via ``ConntrailConfig.model_prices`` or the
    ``CONNTRAIL_PRICE_OVERRIDES`` env var (JSON: {"<prefix>": [in, cached, out]}).
    Prices drift — treat every cost_usd as an estimate, not an invoice.

Cost math follows the provider conventions surfaced through langchain-core:
``input_tokens`` INCLUDES cached and cache-written tokens, cache reads are
billed at the discounted cached rate, and cache writes (Anthropic only,
1.25x premium) are billed at the write multiplier.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig, var_child_runnable_config

# Approximate USD per 1M tokens: (input, cached_input, output).
# Longest-prefix match wins; see resolve_price(). Estimates only.
_MODEL_PRICES: dict[str, tuple[float, float, float]] = {
    # Anthropic (cache read ~0.1x, cache write ~1.25x input)
    "claude-opus": (15.00, 1.50, 75.00),
    "claude-sonnet": (3.00, 0.30, 15.00),
    "claude-haiku": (1.00, 0.10, 5.00),
    "claude-3-5-haiku": (0.80, 0.08, 4.00),
    "claude-3-5-sonnet": (3.00, 0.30, 15.00),
    # OpenAI (cache read 50-90% off depending on family)
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-5": (1.25, 0.125, 10.00),
    "o4-mini": (1.10, 0.55, 4.40),
    "o3": (2.00, 1.00, 8.00),
    "o1": (15.00, 7.50, 60.00),
    # Google Gemini (flash-family default; explicit 2.5-pro entry)
    "gemini-2.5-pro": (1.25, 0.31, 10.00),
    "gemini": (0.10, 0.025, 0.40),
    # Groq
    "llama-3.1-8b": (0.05, 0.00, 0.08),
    "llama-3.3-70b": (0.59, 0.00, 0.79),
}

# Fallback for models not in the table (gpt-4o-class; conservative middle ground).
_DEFAULT_PRICE: tuple[float, float, float] = (2.50, 1.25, 10.00)
_LOCAL_PRICE: tuple[float, float, float] = (0.00, 0.00, 0.00)

# Anthropic charges a 1.25x write premium on cache-creation tokens.
_CACHE_WRITE_MULTIPLIER = 1.25

# Vendor path prefixes stripped before price lookup ("openrouter/anthropic/claude-..."
# resolves like a bare "claude-..."; Google names its models "models/gemini-...").
_VENDOR_PREFIXES = (
    "openrouter/",
    "anthropic/",
    "openai/",
    "google/",
    "models/",
    "meta-llama/",
    "mistralai/",
    "qwen/",
    "deepseek/",
    "x-ai/",
    "amazon/",
)

# Minimum normalized characters for an instruction block to be fingerprinted.
_MIN_BLOCK_CHARS = 120


# --------------------------------------------------------------------------
# Token usage
# --------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Normalized token counts for one or more LLM calls.

    ``input_tokens`` includes cached and cache-written tokens (langchain-core
    convention); ``cached_input_tokens`` / ``cache_write_tokens`` break out
    the cache-read / cache-creation subsets of it.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def merged(cls, usages: Iterable[TokenUsage]) -> TokenUsage | None:
        """Merge usages; None if the iterable is empty."""
        usages = list(usages)
        if not usages:
            return None
        return cls(
            input_tokens=sum(u.input_tokens for u in usages),
            output_tokens=sum(u.output_tokens for u in usages),
            cached_input_tokens=sum(u.cached_input_tokens for u in usages),
            cache_write_tokens=sum(u.cache_write_tokens for u in usages),
        )


def usage_from_message(message: Any) -> TokenUsage | None:
    """Extract normalized token usage from a LangChain AIMessage-like object.

    Prefers ``usage_metadata`` (langchain-core normalization, provider-agnostic),
    falls back to provider ``response_metadata`` shapes (Anthropic ``usage``,
    OpenAI/Groq ``token_usage``). Returns None when no usage is reported.
    """
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and usage.get("input_tokens") is not None:
        details = usage.get("input_token_details") or {}
        return TokenUsage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cached_input_tokens=int(details.get("cache_read") or 0),
            cache_write_tokens=int(details.get("cache_creation") or 0),
        )

    rm = getattr(message, "response_metadata", None) or {}
    if not isinstance(rm, dict):
        return None

    # Anthropic shape: input_tokens EXCLUDES cached tokens (added here).
    au = rm.get("usage")
    if isinstance(au, dict) and (au.get("input_tokens") or au.get("output_tokens")):
        cached = int(au.get("cache_read_input_tokens") or 0)
        written = int(au.get("cache_creation_input_tokens") or 0)
        return TokenUsage(
            input_tokens=int(au.get("input_tokens") or 0) + cached + written,
            output_tokens=int(au.get("output_tokens") or 0),
            cached_input_tokens=cached,
            cache_write_tokens=written,
        )

    # OpenAI / Groq shape: prompt_tokens INCLUDES cached tokens.
    tu = rm.get("token_usage")
    if isinstance(tu, dict) and (tu.get("prompt_tokens") or tu.get("completion_tokens")):
        details = tu.get("prompt_tokens_details") or {}
        return TokenUsage(
            input_tokens=int(tu.get("prompt_tokens") or 0),
            output_tokens=int(tu.get("completion_tokens") or 0),
            cached_input_tokens=int(details.get("cached_tokens") or 0),
        )

    return None


def model_name_from_message(message: Any) -> str:
    """Best-effort model name from a LangChain AIMessage-like object."""
    rm = getattr(message, "response_metadata", None) or {}
    if isinstance(rm, dict):
        for key in ("model_name", "model", "model_id"):
            value = rm.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) when no tokenizer is available."""
    return max(1, len(text) // 4) if text else 0


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


def normalize_model_name(model: str) -> str:
    """Strip vendor path prefixes; map local/* to the literal "local"."""
    name = (model or "").strip()
    stripped = True
    while stripped:
        stripped = False
        low = name.lower()
        for prefix in _VENDOR_PREFIXES:
            if low.startswith(prefix):
                name = name[len(prefix):]
                stripped = True
                break
    name = name.strip()
    if name.lower() == "local" or name.lower().startswith("local/"):
        return "local"
    return name


def resolve_price(
    model: str,
    price_overrides: Mapping[str, tuple[float, float, float]] | None = None,
) -> tuple[float, float, float]:
    """Resolve (input, cached_input, output) USD-per-1M prices for a model.

    Longest-prefix match; overrides win over the built-in table. ``local``
    models are free; empty/unknown names fall back to the default price.
    """
    name = normalize_model_name(model)
    if name == "local":
        return _LOCAL_PRICE
    if not name:
        return _DEFAULT_PRICE
    for table in (price_overrides, _MODEL_PRICES):
        if not table:
            continue
        for prefix in sorted(table, key=len, reverse=True):
            if name.startswith(prefix):
                price = table[prefix]
                return (float(price[0]), float(price[1]), float(price[2]))
    return _DEFAULT_PRICE


def estimate_cost(
    model: str,
    usage: TokenUsage | None,
    price_overrides: Mapping[str, tuple[float, float, float]] | None = None,
) -> float:
    """Estimate USD cost of a usage record on a given model (0.0 if no usage)."""
    if usage is None:
        return 0.0
    input_price, cached_price, output_price = resolve_price(model, price_overrides)
    uncached = max(
        0, usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens
    )
    cost = (
        uncached * input_price
        + usage.cached_input_tokens * cached_price
        + usage.cache_write_tokens * input_price * _CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * output_price
    ) / 1_000_000
    return round(cost, 6)


def price_overrides_from_env(
    env: Mapping[str, str] | None = None,
) -> dict[str, tuple[float, float, float]] | None:
    """Parse CONNTRAIL_PRICE_OVERRIDES (JSON {"<prefix>": [in, cached, out]})."""
    raw = (env if env is not None else os.environ).get("CONNTRAIL_PRICE_OVERRIDES", "")
    if not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"CONNTRAIL_PRICE_OVERRIDES is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("CONNTRAIL_PRICE_OVERRIDES must be a JSON object of prefix -> prices")
    overrides: dict[str, tuple[float, float, float]] = {}
    for prefix, price in data.items():
        if not isinstance(price, (list, tuple)) or len(price) != 3:
            raise ValueError(
                f"CONNTRAIL_PRICE_OVERRIDES[{prefix!r}] must be [input, cached, output]"
            )
        overrides[str(prefix)] = (float(price[0]), float(price[1]), float(price[2]))
    return overrides


# --------------------------------------------------------------------------
# Prompt fingerprinting
# --------------------------------------------------------------------------


def _message_text(message: Any) -> str:
    """Extract text from a message content (str or content-block list)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _is_system_message(message: Any) -> bool:
    if isinstance(message, SystemMessage):
        return True
    return getattr(message, "type", "") == "system"


def _normalize_block(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _block_hashes(text: str) -> list[str]:
    """Fingerprint a prompt: hash of the whole text plus each large block.

    Blocks are paragraph-separated chunks at least _MIN_BLOCK_CHARS long
    (after whitespace normalization). Short texts are skipped entirely.
    """
    hashes: list[str] = []

    def _add(chunk: str) -> None:
        normalized = _normalize_block(chunk)
        if len(normalized) >= _MIN_BLOCK_CHARS:
            digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]
            if digest not in hashes:
                hashes.append(digest)

    _add(text)
    for block in re.split(r"\n\s*\n", text):
        _add(block)
    return hashes


# --------------------------------------------------------------------------
# Callback handler
# --------------------------------------------------------------------------


@dataclass
class LLMCallRecord:
    """One LLM call observed inside a traced node run."""

    model: str = ""
    usage: TokenUsage | None = None
    prompt_chars: int = 0
    system_prompt_chars: int = 0
    message_count: int = 0
    prompt_hashes: list[str] = field(default_factory=list)


@dataclass
class _PromptStats:
    prompt_chars: int = 0
    system_prompt_chars: int = 0
    message_count: int = 0
    prompt_hashes: list[str] = field(default_factory=list)
    model: str = ""


def _flatten_messages(messages: Any) -> list:
    """Flatten on_chat_model_start's messages arg (list of message batches)."""
    items = messages if isinstance(messages, (list, tuple)) else [messages]
    flat: list = []
    for item in items:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _model_from_start(serialized: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> str:
    if isinstance(serialized, dict):
        name = serialized.get("name") or serialized.get("id")
        if isinstance(name, str) and name:
            return name
        ids = serialized.get("id")
        if isinstance(ids, list) and ids and isinstance(ids[-1], str):
            return ids[-1]
    params = kwargs.get("invocation_params")
    if isinstance(params, dict):
        for key in ("model_name", "model"):
            value = params.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


class CostCallbackHandler(BaseCallbackHandler):
    """Aggregates LLM-call cost data for everything run under it.

    Injected around node calls via llm_cost_capture(); observes every
    LangChain chat-model call made inside that scope — token usage, cache
    hits, prompt sizes, and instruction-block fingerprints (never raw
    prompt text).
    """

    def __init__(self) -> None:
        self.calls: list[LLMCallRecord] = []
        self._prompt_stats: dict[UUID, _PromptStats] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        flat = _flatten_messages(messages)
        stats = _PromptStats(message_count=len(flat))
        hashes: list[str] = []
        for message in flat:
            text = _message_text(message)
            stats.prompt_chars += len(text)
            if _is_system_message(message):
                stats.system_prompt_chars += len(text)
                hashes.extend(_block_hashes(text))
        stats.prompt_hashes = list(dict.fromkeys(hashes))
        stats.model = _model_from_start(serialized, kwargs)
        self._prompt_stats[run_id] = stats

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        message = None
        generations = getattr(response, "generations", None) or []
        for batch in generations:
            for generation in batch or []:
                message = getattr(generation, "message", None)
                if message is not None:
                    break
            if message is not None:
                break
        usage = usage_from_message(message) if message is not None else None
        model = (model_name_from_message(message) if message is not None else "") or (
            self._prompt_stats.get(run_id).model if run_id in self._prompt_stats else ""
        )
        stats = self._prompt_stats.pop(run_id, _PromptStats())
        self.calls.append(
            LLMCallRecord(
                model=model,
                usage=usage,
                prompt_chars=stats.prompt_chars,
                system_prompt_chars=stats.system_prompt_chars,
                message_count=stats.message_count,
                prompt_hashes=stats.prompt_hashes,
            )
        )

    # -- aggregation ------------------------------------------------------

    def total_usage(self) -> TokenUsage | None:
        """Merged usage across observed calls that reported usage."""
        return TokenUsage.merged(c.usage for c in self.calls if c.usage is not None)

    def usage_summary(self) -> dict[str, Any] | None:
        """JSON-safe cost summary; None when no LLM calls were observed."""
        if not self.calls:
            return None
        total = self.total_usage()
        prompt_chars = sum(c.prompt_chars for c in self.calls)
        hashes = sorted({h for c in self.calls for h in c.prompt_hashes})
        # Hash occurrences across calls minus unique hashes: > 0 means the
        # same instruction block was sent on multiple LLM calls.
        hash_occurrences = sum(len(set(c.prompt_hashes)) for c in self.calls)
        return {
            "llm_call_count": len(self.calls),
            "input_tokens": total.input_tokens if total else 0,
            "output_tokens": total.output_tokens if total else 0,
            "cached_input_tokens": total.cached_input_tokens if total else 0,
            "cache_write_tokens": total.cache_write_tokens if total else 0,
            "total_tokens": total.total_tokens if total else 0,
            "models": sorted({c.model for c in self.calls if c.model}),
            "prompt_chars": prompt_chars,
            "system_prompt_chars": sum(c.system_prompt_chars for c in self.calls),
            "prompt_token_estimate": estimate_tokens("x" * prompt_chars),
            "prompt_hashes": hashes[:64],
            "repeated_prompt_blocks": max(0, hash_occurrences - len(hashes)),
        }

    def estimated_cost(
        self,
        price_overrides: Mapping[str, tuple[float, float, float]] | None = None,
    ) -> float | None:
        """Per-call cost estimate summed over calls with usage data; None if none."""
        total = 0.0
        seen_usage = False
        for call in self.calls:
            if call.usage is None:
                continue
            seen_usage = True
            total += estimate_cost(call.model, call.usage, price_overrides)
        return round(total, 6) if seen_usage else None


# --------------------------------------------------------------------------
# Capture plumbing
# --------------------------------------------------------------------------


@contextmanager
def llm_cost_capture(handler: CostCallbackHandler) -> Iterator[CostCallbackHandler]:
    """Make LLM calls inside this scope visible to ``handler``.

    Sets the LangChain child-runnable config contextvar (merging with any
    config the host app already set) so chat models invoked with no explicit
    config pick the handler up. Host callback managers are copied — never
    mutated — so concurrent host runs are unaffected. Explicitly-configured
    host calls may bypass the handler — capture is best-effort by design.
    """
    from langchain_core.callbacks import BaseCallbackManager

    existing = var_child_runnable_config.get()
    if existing is None:
        config: RunnableConfig = {"callbacks": [handler]}
    else:
        config = dict(existing)
        callbacks = config.get("callbacks")
        if callbacks is None:
            config["callbacks"] = [handler]
        elif isinstance(callbacks, BaseCallbackManager):
            merged = callbacks.copy()
            merged.add_handler(handler, inherit=False)
            config["callbacks"] = merged
        elif isinstance(callbacks, (list, tuple)):
            merged_list = list(callbacks)
            if handler not in merged_list:
                merged_list.append(handler)
            config["callbacks"] = merged_list
        else:
            # Unrecognized callbacks shape — replace rather than guess.
            config["callbacks"] = [handler]
    token = var_child_runnable_config.set(config)
    try:
        yield handler
    finally:
        var_child_runnable_config.reset(token)


@dataclass
class CostCapture:
    """Hot-path cost data for one node invocation (handler + latency)."""

    handler: CostCallbackHandler = field(default_factory=CostCallbackHandler)
    latency_ms: float | None = None

    def finish(self, elapsed_seconds: float) -> None:
        self.latency_ms = round(elapsed_seconds * 1000, 1)

    def usage_summary(self) -> dict[str, Any] | None:
        return self.handler.usage_summary()

    def estimated_cost(
        self,
        price_overrides: Mapping[str, tuple[float, float, float]] | None = None,
    ) -> float | None:
        return self.handler.estimated_cost(price_overrides)


def build_analysis_overhead(
    *,
    re_run_usage: TokenUsage | None,
    re_run_models: list[str],
    re_run_latency_ms: float | None,
    retries: int,
    contrast_usage: TokenUsage | None,
    contrast_model: str,
    price_overrides: Mapping[str, tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    """Assemble the observer's own per-trace overhead summary.

    Combines the 4 divergence re-runs (node-internal LLM usage, priced with
    the first observed re-run model as an approximation) with the contrast
    generation call (priced with the exact contrast model).
    """
    total = TokenUsage.merged(u for u in (re_run_usage, contrast_usage) if u is not None)
    re_run_cost = (
        estimate_cost(re_run_models[0] if re_run_models else "", re_run_usage, price_overrides)
        if re_run_usage is not None
        else 0.0
    )
    contrast_cost = estimate_cost(contrast_model, contrast_usage, price_overrides)
    return {
        "input_tokens": total.input_tokens if total else 0,
        "output_tokens": total.output_tokens if total else 0,
        "total_tokens": total.total_tokens if total else 0,
        "retries": retries,
        "latency_ms": round(re_run_latency_ms, 1) if re_run_latency_ms is not None else None,
        "cost_usd": round(re_run_cost + contrast_cost, 6),
    }


def now_monotonic() -> float:
    """perf_counter (separate helper so tests can fake timing uniformly)."""
    return time.perf_counter()
