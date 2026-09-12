"""
Tests for provider resolution (conntrail.utils.providers).

Provider SDKs (langchain-groq/-anthropic/-openai) are not base dependencies —
_build_model imports them lazily. Tests inject fake modules into sys.modules
rather than requiring the real packages to be installed.
"""
import json
import sys
import types
import urllib.error

import pytest

from conntrail.utils import providers
from conntrail.utils.providers import (
    _available_provider,
    _build_model,
    _get_local_token,
    _infer_provider,
    _resolve_local_api_key,
    get_chat_model,
)

# ---------------------------------------------------------------------------
# _infer_provider — prefix inference
# ---------------------------------------------------------------------------

class TestInferProvider:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("llama-3.1-8b-instant", "groq"),
            ("mixtral-8x7b", "groq"),
            ("gemma-7b-it", "groq"),
            ("qwen-2.5-72b", "groq"),
            ("deepseek-r1", "groq"),
            ("claude-haiku-4-5-20251001", "anthropic"),
            ("gpt-4o-mini", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("gemini-2.0-flash", "google"),
            ("gemini-2.5-pro", "google"),
            ("o4-mini", "openai"),
        ],
    )
    def test_known_prefixes(self, model, expected):
        assert _infer_provider(model) == expected

    def test_unknown_prefix_returns_none(self):
        assert _infer_provider("some-unknown-model") is None

    def test_local_bare(self):
        assert _infer_provider("local") == "local"

    def test_local_with_name(self):
        assert _infer_provider("local/qwen3") == "local"

    def test_openrouter_bare(self):
        assert _infer_provider("openrouter") == "openrouter"

    def test_openrouter_with_vendor_model(self):
        assert _infer_provider("openrouter/anthropic/claude-3-haiku") == "openrouter"

    def test_prefix_case_insensitive(self):
        assert _infer_provider("CLAUDE-haiku") == "anthropic"


# ---------------------------------------------------------------------------
# _available_provider — fallback priority
# ---------------------------------------------------------------------------

class TestAvailableProvider:
    def _clear_all_keys(self, monkeypatch):
        for key in (
            "GROQ_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_prefers_groq_first(self, monkeypatch):
        self._clear_all_keys(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        assert _available_provider() == "groq"

    def test_falls_back_to_anthropic(self, monkeypatch):
        self._clear_all_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        assert _available_provider() == "anthropic"

    def test_falls_back_to_openai(self, monkeypatch):
        self._clear_all_keys(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        assert _available_provider() == "openai"

    def test_falls_back_to_google(self, monkeypatch):
        self._clear_all_keys(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "gg")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or")
        assert _available_provider() == "google"

    def test_falls_back_to_openrouter(self, monkeypatch):
        self._clear_all_keys(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or")
        assert _available_provider() == "openrouter"

    def test_raises_when_no_key_set(self, monkeypatch):
        self._clear_all_keys(monkeypatch)
        with pytest.raises(EnvironmentError, match="No LLM API key found"):
            _available_provider()


# ---------------------------------------------------------------------------
# get_chat_model — resolution + fallback + local/openrouter routing
# ---------------------------------------------------------------------------

class TestGetChatModel:
    def test_resolves_inferred_provider_when_key_present(self, monkeypatch, mocker):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
        spy = mocker.patch.object(providers, "_build_model")
        get_chat_model("claude-haiku-4-5-20251001", max_tokens=100)
        spy.assert_called_once_with(
            "anthropic", "claude-haiku-4-5-20251001", max_tokens=100, temperature=0.0
        )

    def test_falls_back_when_inferred_provider_key_missing(self, monkeypatch, mocker):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "g")
        spy = mocker.patch.object(providers, "_build_model")
        get_chat_model("claude-haiku-4-5-20251001")
        spy.assert_called_once_with(
            "groq", "llama-3.1-8b-instant", max_tokens=512, temperature=0.0
        )

    def test_unknown_prefix_falls_back_to_available_provider(self, monkeypatch, mocker):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        spy = mocker.patch.object(providers, "_build_model")
        get_chat_model("some-unknown-model")
        spy.assert_called_once_with(
            "openai", "gpt-4o-mini", max_tokens=512, temperature=0.0
        )

    def test_resolves_gemini_when_key_present(self, monkeypatch, mocker):
        monkeypatch.setenv("GOOGLE_API_KEY", "gg")
        spy = mocker.patch.object(providers, "_build_model")
        get_chat_model("gemini-2.0-flash", max_tokens=100)
        spy.assert_called_once_with(
            "google", "gemini-2.0-flash", max_tokens=100, temperature=0.0
        )

    def test_local_bare_uses_default_model_name(self, mocker):
        spy = mocker.patch.object(providers, "_build_model")
        get_chat_model("local")
        spy.assert_called_once_with(
            "local", providers._LOCAL_MODEL_NAME, max_tokens=512, temperature=0.0
        )

    def test_local_with_name_routes_to_local_server(self, mocker):
        spy = mocker.patch.object(providers, "_build_model")
        get_chat_model("local/qwen3")
        spy.assert_called_once_with("local", "qwen3", max_tokens=512, temperature=0.0)

    def test_openrouter_bare_uses_fallback_model(self, mocker):
        spy = mocker.patch.object(providers, "_build_model")
        get_chat_model("openrouter")
        spy.assert_called_once_with(
            "openrouter", providers._FALLBACK_MODELS["openrouter"], max_tokens=512, temperature=0.0
        )

    def test_openrouter_with_vendor_model_routes_correctly(self, mocker):
        spy = mocker.patch.object(providers, "_build_model")
        get_chat_model("openrouter/anthropic/claude-3-haiku")
        spy.assert_called_once_with(
            "openrouter", "anthropic/claude-3-haiku", max_tokens=512, temperature=0.0
        )


# ---------------------------------------------------------------------------
# _resolve_local_api_key — auth-mode dispatch
# ---------------------------------------------------------------------------

class TestResolveLocalApiKey:
    def test_none_mode_returns_placeholder(self, monkeypatch):
        monkeypatch.setattr(providers, "_LOCAL_AUTH_MODE", "none")
        assert _resolve_local_api_key() == "not-needed"

    def test_api_key_mode_returns_local_api_key(self, monkeypatch):
        monkeypatch.setattr(providers, "_LOCAL_AUTH_MODE", "api_key")
        monkeypatch.setenv("LOCAL_API_KEY", "static-token")
        assert _resolve_local_api_key() == "static-token"

    def test_jwt_mode_exchanges_token(self, monkeypatch, mocker):
        monkeypatch.setattr(providers, "_LOCAL_AUTH_MODE", "jwt")
        mocker.patch.object(providers, "_get_local_token", return_value="tok123")
        assert _resolve_local_api_key() == "tok123"


# ---------------------------------------------------------------------------
# _build_model — dispatch to the right LangChain class (fake modules injected)
# ---------------------------------------------------------------------------

class TestBuildModel:
    def _inject_fake_module(self, monkeypatch, module_name, class_name):
        fake_module = types.ModuleType(module_name)
        fake_class = type(class_name, (), {"__init__": lambda self, **kw: setattr(self, "kwargs", kw)})
        setattr(fake_module, class_name, fake_class)
        monkeypatch.setitem(sys.modules, module_name, fake_module)
        return fake_class

    def test_groq_dispatch(self, monkeypatch):
        fake_class = self._inject_fake_module(monkeypatch, "langchain_groq", "ChatGroq")
        model = _build_model("groq", "llama-3.1-8b-instant", max_tokens=50, temperature=0.1)
        assert isinstance(model, fake_class)
        assert model.kwargs["model"] == "llama-3.1-8b-instant"

    def test_anthropic_dispatch(self, monkeypatch):
        fake_class = self._inject_fake_module(monkeypatch, "langchain_anthropic", "ChatAnthropic")
        model = _build_model("anthropic", "claude-haiku-4-5-20251001", max_tokens=50, temperature=0.0)
        assert isinstance(model, fake_class)

    def test_openai_dispatch(self, monkeypatch):
        fake_class = self._inject_fake_module(monkeypatch, "langchain_openai", "ChatOpenAI")
        model = _build_model("openai", "gpt-4o-mini", max_tokens=50, temperature=0.0)
        assert isinstance(model, fake_class)
        # Usage must be requested explicitly — langchain-openai auto-enables it
        # only for the default api.openai.com base URL.
        assert model.kwargs["stream_usage"] is True

    def test_openrouter_dispatch(self, monkeypatch):
        fake_class = self._inject_fake_module(monkeypatch, "langchain_openai", "ChatOpenAI")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        model = _build_model("openrouter", "anthropic/claude-3-haiku", max_tokens=50, temperature=0.0)
        assert isinstance(model, fake_class)
        assert model.kwargs["base_url"] == providers._OPENROUTER_BASE_URL
        assert model.kwargs["api_key"] == "or-key"
        assert model.kwargs["stream_usage"] is True

    def test_gemini_dispatch(self, monkeypatch):
        fake_class = self._inject_fake_module(
            monkeypatch, "langchain_google_genai", "ChatGoogleGenerativeAI"
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
        model = _build_model("google", "gemini-2.0-flash", max_tokens=50, temperature=0.0)
        assert isinstance(model, fake_class)
        assert model.kwargs["model"] == "gemini-2.0-flash"
        assert model.kwargs["google_api_key"] == "g-key"
        assert model.kwargs["max_output_tokens"] == 50

    def test_gemini_key_falls_back_to_gemini_api_key_env(self, monkeypatch):
        self._inject_fake_module(
            monkeypatch, "langchain_google_genai", "ChatGoogleGenerativeAI"
        )
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "alt-key")
        model = _build_model("google", "gemini-2.0-flash", max_tokens=50, temperature=0.0)
        assert model.kwargs["google_api_key"] == "alt-key"

    def test_local_dispatch_none_auth_no_streaming_hack(self, monkeypatch):
        fake_class = self._inject_fake_module(monkeypatch, "langchain_openai", "ChatOpenAI")
        monkeypatch.setattr(providers, "_LOCAL_AUTH_MODE", "none")
        model = _build_model("local", "some-model", max_tokens=50, temperature=0.0)
        assert isinstance(model, fake_class)
        assert model.kwargs["api_key"] == "not-needed"
        assert "streaming" not in model.kwargs
        # Local OpenAI-compatible servers (Unsloth, llama.cpp, Ollama /v1)
        # report usage only when asked.
        assert model.kwargs["stream_usage"] is True

    def test_local_dispatch_api_key_auth(self, monkeypatch):
        fake_class = self._inject_fake_module(monkeypatch, "langchain_openai", "ChatOpenAI")
        monkeypatch.setattr(providers, "_LOCAL_AUTH_MODE", "api_key")
        monkeypatch.setenv("LOCAL_API_KEY", "static-token")
        model = _build_model("local", "some-model", max_tokens=50, temperature=0.0)
        assert isinstance(model, fake_class)
        assert model.kwargs["api_key"] == "static-token"
        assert model.kwargs["stream_usage"] is True

    def test_local_dispatch_jwt_auth_fetches_token(self, monkeypatch, mocker):
        fake_class = self._inject_fake_module(monkeypatch, "langchain_openai", "ChatOpenAI")
        monkeypatch.setattr(providers, "_LOCAL_AUTH_MODE", "jwt")
        mocker.patch.object(providers, "_get_local_token", return_value="tok123")
        model = _build_model("local", "qwen3", max_tokens=50, temperature=0.0)
        assert isinstance(model, fake_class)
        assert model.kwargs["api_key"] == "tok123"
        assert model.kwargs["streaming"] is True
        assert model.kwargs["stream_usage"] is True

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            _build_model("bogus", "some-model", max_tokens=50)


# ---------------------------------------------------------------------------
# _get_local_token — JWT exchange for the local server (urllib mocked)
# ---------------------------------------------------------------------------

class TestGetLocalToken:
    def test_missing_password_raises(self, monkeypatch):
        monkeypatch.delenv("LOCAL_PASSWORD", raising=False)
        monkeypatch.setattr(providers, "_LOCAL_PASSWORD", "")
        with pytest.raises(EnvironmentError, match="LOCAL_PASSWORD"):
            _get_local_token()

    def test_successful_token_exchange(self, monkeypatch, mocker):
        monkeypatch.setenv("LOCAL_PASSWORD", "secret")

        fake_response = mocker.MagicMock()
        fake_response.read.return_value = json.dumps({"access_token": "abc123"}).encode()
        fake_response.__enter__ = mocker.MagicMock(return_value=fake_response)
        fake_response.__exit__ = mocker.MagicMock(return_value=False)
        mocker.patch("urllib.request.urlopen", return_value=fake_response)

        token = _get_local_token()
        assert token == "abc123"

    def test_missing_access_token_in_response_raises(self, monkeypatch, mocker):
        monkeypatch.setenv("LOCAL_PASSWORD", "secret")

        fake_response = mocker.MagicMock()
        fake_response.read.return_value = json.dumps({"nope": "no token here"}).encode()
        fake_response.__enter__ = mocker.MagicMock(return_value=fake_response)
        fake_response.__exit__ = mocker.MagicMock(return_value=False)
        mocker.patch("urllib.request.urlopen", return_value=fake_response)

        with pytest.raises(EnvironmentError, match="No access_token"):
            _get_local_token()

    def test_http_error_raises_environment_error(self, monkeypatch, mocker):
        monkeypatch.setenv("LOCAL_PASSWORD", "secret")

        def raise_http_error(*args, **kwargs):
            raise urllib.error.HTTPError(
                url="http://x", code=401, msg="unauthorized", hdrs=None, fp=mocker.MagicMock(read=lambda: b"bad creds")
            )

        mocker.patch("urllib.request.urlopen", side_effect=raise_http_error)

        with pytest.raises(EnvironmentError, match="login failed"):
            _get_local_token()
