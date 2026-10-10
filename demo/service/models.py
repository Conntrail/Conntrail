"""
Model resolution for the live demo.

The server's own free-tier key is used for the capped trial; a visitor can
bring their own free key instead. Only free providers are supported on purpose
— the demo must never incur per-token cost at scale.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Free-tier defaults. Groq has the highest published daily request cap; the
# OpenRouter entries are ":free" variants (the model host pays, and typically
# uses the data). Keep these to models that are cheap, fast and non-reasoning.
FREE_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "cohere/north-mini-code:free",
}

# Fallbacks tried in order when a free model is rate-limited (429) or returns
# an unusable response. All are OpenRouter ":free" variants.
FREE_FALLBACKS = {
    "groq": ["llama-3.1-8b-instant"],
    "openrouter": [
        "poolside/laguna-s-2.1:free",
        "dots-studio/dots-3-note-preview:free",
        "google/gemma-4-31b-it:free",
    ],
}

# Preference order for the server's own trial key.
_SERVER_ORDER = ("groq", "openrouter")

_KEY_ENV = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


class NoProviderKey(RuntimeError):
    """No usable free-tier key is configured on the server."""


@dataclass
class ModelConfig:
    provider: str
    model: str
    api_key: str


def infer_provider_from_key(api_key: str) -> str | None:
    """Best-effort provider inference from a key's prefix."""
    if api_key.startswith("sk-or-"):
        return "openrouter"
    if api_key.startswith("gsk_"):
        return "groq"
    return None


def server_config() -> ModelConfig | None:
    """The server's own free-tier config, or None if no key is set."""
    for provider in _SERVER_ORDER:
        key = os.getenv(_KEY_ENV[provider])
        if key:
            model = os.getenv(f"DEMO_MODEL_{provider.upper()}", FREE_MODELS[provider])
            return ModelConfig(provider=provider, model=model, api_key=key)
    return None


def resolve_config(provider: str | None = None, api_key: str | None = None) -> ModelConfig:
    """Resolve the model config for a request.

    A supplied ``api_key`` (bring-your-own-key) wins; otherwise the server's
    free-tier key is used. Raises NoProviderKey when neither is available.
    """
    if api_key:
        resolved = provider or infer_provider_from_key(api_key) or os.getenv("DEMO_PROVIDER") or "groq"
        if resolved not in FREE_MODELS:
            raise NoProviderKey(f"unsupported provider: {resolved}")
        model = os.getenv(f"DEMO_MODEL_{resolved.upper()}", FREE_MODELS[resolved])
        return ModelConfig(provider=resolved, model=model, api_key=api_key)

    cfg = server_config()
    if cfg is None:
        raise NoProviderKey(
            "No free-tier key configured. Set GROQ_API_KEY or OPENROUTER_API_KEY on the server."
        )
    return cfg


def candidate_configs(
    provider: str | None = None, api_key: str | None = None
) -> list[ModelConfig]:
    """Primary config first, then free fallbacks (used on 429 / unusable output)."""
    primary = resolve_config(provider, api_key)
    ordered = [primary.model, *FREE_FALLBACKS.get(primary.provider, [])]
    seen: set[str] = set()
    configs: list[ModelConfig] = []
    for model in ordered:
        if model in seen:
            continue
        seen.add(model)
        configs.append(ModelConfig(provider=primary.provider, model=model, api_key=primary.api_key))
    return configs


def build_chat_model(cfg: ModelConfig, *, max_tokens: int = 1500):
    """Construct a LangChain chat model for the resolved provider."""
    if cfg.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=cfg.model,
            api_key=cfg.api_key,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    if cfg.provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=max_tokens,
            stream_usage=True,
        )
    raise NoProviderKey(f"unsupported provider: {cfg.provider}")
