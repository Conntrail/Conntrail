"""
Tests for the Sequence-game tracing harness's pure helpers.

The tournament repo is external, so these cover everything that does not need
it: priority parsing and provider/path resolution. The full traced run is an
example, exercised live against the local server / OpenRouter.
"""
from __future__ import annotations

import pytest

from conntrail.utils import providers
from examples.sequence.trace_sequence_agent import (
    PRIORITIES,
    native_target,
    parse_priority,
    resolve_game_repo,
)

# ---------------------------------------------------------------------------
# parse_priority
# ---------------------------------------------------------------------------


class TestParsePriority:
    def test_parses_json_answer(self):
        assert parse_priority('{"priority": "BLOCK"}') == "BLOCK"

    def test_parses_bare_word(self):
        assert parse_priority("I would choose CENTER.") == "CENTER"

    def test_case_insensitive(self):
        assert parse_priority("advance") == "ADVANCE"

    def test_strips_inline_think_block(self):
        # A reasoning model may mention priorities before its real answer.
        text = "<" + "think>FORK seems tempting.</think>WIN"
        assert parse_priority(text) == "WIN"

    def test_first_label_wins(self):
        assert parse_priority("WIN, or maybe BLOCK") == "WIN"

    def test_unknown_when_no_label(self):
        assert parse_priority("I am not sure what to do") == "unknown"

    def test_empty_and_none(self):
        assert parse_priority("") == "unknown"
        assert parse_priority(None) == "unknown"

    def test_all_priorities_recognised(self):
        for label in PRIORITIES:
            assert parse_priority(f"priority: {label}") == label


# ---------------------------------------------------------------------------
# native_target
# ---------------------------------------------------------------------------


class TestNativeTarget:
    def test_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        base_url, api_key, api_model = native_target("openrouter/deepseek/deepseek-v4.1-flash")
        assert base_url == "https://openrouter.ai/api/v1"
        assert api_key == "sk-or-test"
        assert api_model == "deepseek/deepseek-v4.1-flash"  # prefix stripped for the API

    def test_openrouter_requires_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
            native_target("openrouter/openai/gpt-4o-mini")

    def test_local_uses_server_url_and_name(self, monkeypatch):
        monkeypatch.setattr(providers, "_LOCAL_AUTH_MODE", "none")
        monkeypatch.setenv("LOCAL_LLM_URL", "http://127.0.0.1:11434/v1")
        base_url, api_key, api_model = native_target("local/unsloth/gemma-4-12b-it-GGUF")
        assert base_url == "http://127.0.0.1:11434/v1"
        assert api_key == "not-needed"
        assert api_model == "unsloth/gemma-4-12b-it-GGUF"

    def test_unsupported_model_for_native_flavor(self):
        with pytest.raises(SystemExit, match="skip-native"):
            native_target("gemini-2.0-flash")


# ---------------------------------------------------------------------------
# resolve_game_repo
# ---------------------------------------------------------------------------


class TestResolveGameRepo:
    def test_explicit_path(self, tmp_path):
        (tmp_path / "AI").mkdir()
        (tmp_path / "AI" / "prompt.py").write_text("")
        assert resolve_game_repo(str(tmp_path)) == tmp_path

    def test_explicit_path_must_look_like_the_repo(self, tmp_path):
        with pytest.raises(SystemExit, match="does not look like"):
            resolve_game_repo(str(tmp_path))

    def test_env_var(self, monkeypatch, tmp_path):
        (tmp_path / "AI").mkdir()
        (tmp_path / "AI" / "prompt.py").write_text("")
        monkeypatch.setenv("SEQUENCE_GAME_REPO", str(tmp_path))
        assert resolve_game_repo(None) == tmp_path

    def test_missing_everywhere_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SEQUENCE_GAME_REPO", raising=False)
        monkeypatch.setattr(
            "examples.sequence.trace_sequence_agent._GAME_REPO_CANDIDATES", (tmp_path / "nope",)
        )
        with pytest.raises(SystemExit, match="Could not locate"):
            resolve_game_repo(None)
