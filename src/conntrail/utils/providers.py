"""
Provider resolution — maps model strings to LangChain chat model instances.

Supports Groq, Anthropic, OpenAI, OpenRouter, and local OpenAI-compatible
servers (Ollama, llama.cpp, vLLM, Unsloth Studio, etc.).

Local model usage:
    Pass model="local/<name>" or model="local" to route to the local server.
    Server URL is read from LOCAL_LLM_URL (default: http://127.0.0.1:8888/v1).
    Model name sent to the server is the part after "local/" or LOCAL_MODEL_NAME env var.

    Most local OpenAI-compatible servers (Ollama, llama.cpp, vLLM) don't need
    auth, or accept a static bearer token. A few (e.g. Unsloth Studio) require
    a username/password JWT exchange instead. LOCAL_AUTH_MODE selects which:
      "none"    (default) — no Authorization header sent.
      "api_key" — static bearer token from LOCAL_API_KEY.
      "jwt"     — exchange LOCAL_USERNAME/LOCAL_PASSWORD for a fresh JWT on
                  every call (never cached, so a server restart never causes
                  a run to fail with a stale 401).

OpenRouter usage:
    Pass model="openrouter/<vendor>/<model>" (OpenRouter's own naming, e.g.
    "openrouter/anthropic/claude-3-haiku") or bare "openrouter" to use the
    configured fallback model. Requires OPENROUTER_API_KEY.
"""
from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

# Default local server (OpenAI-compatible: Ollama, llama.cpp, vLLM, Unsloth Studio, ...)
_LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8888/v1")
_LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "unsloth/Qwen3.6-27B-MTP-GGUF")
_LOCAL_AUTH_MODE = os.getenv("LOCAL_AUTH_MODE", "none")  # "none" | "api_key" | "jwt"
_LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "")
_LOCAL_USERNAME = os.getenv("LOCAL_USERNAME", "")
_LOCAL_PASSWORD = os.getenv("LOCAL_PASSWORD", "")

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _get_local_token() -> str:
    """Exchange username/password for a fresh JWT (LOCAL_AUTH_MODE="jwt").

    Never cached — always re-authenticates so a server restart or session
    invalidation never causes a run to fail with stale 401 errors.
    """
    import json as _json
    import urllib.error
    import urllib.request

    base = _LOCAL_LLM_URL.rstrip("/v1").rstrip("/")
    password = os.getenv("LOCAL_PASSWORD") or _LOCAL_PASSWORD
    if not password:
        raise OSError(
            "LOCAL_PASSWORD env var is required when LOCAL_AUTH_MODE=jwt."
        )
    username = os.getenv("LOCAL_USERNAME") or _LOCAL_USERNAME

    payload = _json.dumps({"username": username, "password": password}).encode()
    req = urllib.request.Request(
        f"{base}/api/auth/login",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise OSError(f"Local server login failed ({e.code}): {e.read().decode()}") from e

    token = data.get("access_token")
    if not token:
        raise OSError(f"No access_token in login response: {data}")
    return token


def _resolve_local_api_key() -> str:
    """Resolve the credential to send to the local server, per LOCAL_AUTH_MODE."""
    if _LOCAL_AUTH_MODE == "jwt":
        return _get_local_token()
    if _LOCAL_AUTH_MODE == "api_key":
        return os.getenv("LOCAL_API_KEY") or _LOCAL_API_KEY
    # "none" — most local OpenAI-compatible servers ignore the key entirely,
    # but the client still requires a non-empty placeholder string.
    return "not-needed"


def local_chat_kwargs() -> dict:
    """Extra kwargs an OpenAI-compatible client needs for the local server.

    Re-exports the per-mode decisions _build_model() makes for LangChain
    (JWT mode ⇒ Unsloth Studio ⇒ disable chain-of-thought so small
    max_tokens budgets aren't consumed by reasoning), so non-LangChain
    consumers (e.g. dspy.LM in examples/gepa) can build an equivalent
    client against the same server.
    """
    if _LOCAL_AUTH_MODE == "jwt":
        return {"extra_body": {"enable_thinking": False}}
    return {}


# Model name prefixes → provider
_PREFIX_MAP = {
    "llama": "groq",
    "mixtral": "groq",
    "gemma": "groq",
    "qwen": "groq",
    "deepseek": "groq",
    "claude": "anthropic",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
}

# Cheapest model per provider (used when falling back)
_FALLBACK_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "openrouter": "openai/gpt-4o-mini",
    "local": _LOCAL_MODEL_NAME,
}

_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _infer_provider(model: str) -> str | None:
    # "local" or "local/<model-name>" always routes to local server
    if model == "local" or model.startswith("local/"):
        return "local"
    # "openrouter" or "openrouter/<vendor>/<model>" always routes to OpenRouter
    if model == "openrouter" or model.startswith("openrouter/"):
        return "openrouter"
    prefix = model.split("-")[0].lower()
    return _PREFIX_MAP.get(prefix)


def _available_provider() -> str:
    """Return the first provider with an available API key."""
    for provider in ("groq", "anthropic", "openai", "openrouter"):
        if os.getenv(_KEY_ENV[provider]):
            return provider
    raise OSError(
        "No LLM API key found. Set GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "or OPENROUTER_API_KEY."
    )


def get_chat_model(
    model: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Resolve a model string to a LangChain BaseChatModel instance.

    Special prefix "local" or "local/<name>" routes to the local OpenAI-compatible
    server at LOCAL_LLM_URL (default: http://127.0.0.1:8888/v1).

    Special prefix "openrouter" or "openrouter/<vendor>/<model>" routes to
    OpenRouter's OpenAI-compatible gateway.

    For cloud providers, provider is inferred from the model name prefix. If the
    inferred provider's API key is not set, falls back to the first available
    provider and its default cheap model.

    Args:
        model: Model ID string. Use "local/qwen3" for local inference, or
            "openrouter/anthropic/claude-3-haiku" for OpenRouter.
        max_tokens: Max tokens for the response.
        temperature: Sampling temperature.

    Returns:
        A configured LangChain BaseChatModel.
    """
    inferred = _infer_provider(model)

    if inferred == "local":
        # Extract model name from "local/<name>" or fall back to LOCAL_MODEL_NAME
        local_name = model.split("/", 1)[1] if "/" in model else _LOCAL_MODEL_NAME
        return _build_model("local", local_name, max_tokens=max_tokens, temperature=temperature)

    if inferred == "openrouter":
        # Extract model name from "openrouter/<vendor>/<model>" or use the fallback
        or_name = model.split("/", 1)[1] if "/" in model else _FALLBACK_MODELS["openrouter"]
        return _build_model("openrouter", or_name, max_tokens=max_tokens, temperature=temperature)

    # Check if inferred cloud provider's key is available
    if inferred and os.getenv(_KEY_ENV[inferred]):
        resolved_provider = inferred
        resolved_model = model
    else:
        resolved_provider = _available_provider()
        resolved_model = _FALLBACK_MODELS[resolved_provider]

    return _build_model(resolved_provider, resolved_model, max_tokens=max_tokens, temperature=temperature)


def _build_model(provider: str, model: str, *, max_tokens: int, temperature: float = 0.0) -> BaseChatModel:
    if provider == "local":
        from langchain_openai import ChatOpenAI
        kwargs: dict = {}
        if _LOCAL_AUTH_MODE == "jwt":
            # Unsloth Studio always returns SSE regardless of the stream flag,
            # and disabling Qwen3's chain-of-thought avoids wasting max_tokens on it.
            kwargs = {"streaming": True, "extra_body": {"enable_thinking": False}}
        return ChatOpenAI(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=_LOCAL_LLM_URL,
            api_key=_resolve_local_api_key(),
            **kwargs,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=_OPENROUTER_BASE_URL,
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, max_tokens=max_tokens, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, max_tokens=max_tokens, temperature=temperature)  # type: ignore[call-arg]

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, max_tokens=max_tokens, temperature=temperature)

    raise ValueError(f"Unknown provider: {provider!r}")
