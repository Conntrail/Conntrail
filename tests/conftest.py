"""
Pytest configuration for conntrail tests.

Shared helpers for live-LLM test gating: a live test is runnable when either
a cloud provider key is set OR a local OpenAI-compatible server is explicitly
configured (LOCAL_LLM_URL in the environment, e.g. via .env — Unsloth Studio,
Ollama, llama.cpp, vLLM). The contrast model used by live tests can be
overridden with CONNTRAIL_CONTRAST_MODEL (e.g. "local/unsloth/gemma-4-12b-it-GGUF")
so the whole live pipeline can run against a local server with no cloud keys.
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load repo-root .env so integration tests have API keys available
load_dotenv(Path(__file__).parent.parent / ".env")

_CLOUD_KEY_ENV = ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
_DEFAULT_CONTRAST_MODEL = "claude-haiku-4-5-20251001"


def live_llm_available() -> bool:
    """True when a cloud API key or an explicitly-configured local server is available."""
    if any(os.getenv(k) for k in _CLOUD_KEY_ENV):
        return True
    # An explicitly-set LOCAL_LLM_URL means a local server is configured for
    # live use (the providers module has an implicit default URL, so only an
    # explicit value in the env/.env counts as "intended to be used").
    return bool(os.getenv("LOCAL_LLM_URL"))


def default_contrast_model() -> str:
    """Contrast model for live tests: CONNTRAIL_CONTRAST_MODEL override, else the SDK default."""
    return os.environ.get("CONNTRAIL_CONTRAST_MODEL", _DEFAULT_CONTRAST_MODEL)


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires live API keys")
    config.addinivalue_line("markers", "slow: takes > 10 seconds")


def pytest_sessionfinish(session, exitstatus):
    # Treat "no tests collected" as success on the empty skeleton (no tests
    # exist yet). Once real tests land this branch is never hit.
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
