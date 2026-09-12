"""
Tests for conntrail.cost — token usage extraction, pricing, the LangChain
cost callback handler, and the capture context manager. No API keys needed.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.callbacks import AsyncCallbackManager, BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables.config import var_child_runnable_config

from conntrail.config import ConntrailConfig
from conntrail.cost import (
    CostCallbackHandler,
    CostCapture,
    TokenUsage,
    build_analysis_overhead,
    estimate_cost,
    estimate_tokens,
    llm_cost_capture,
    model_name_from_message,
    normalize_model_name,
    price_overrides_from_env,
    resolve_price,
    usage_from_message,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LONG_BLOCK_A = "You are a customer support router for a large e-commerce platform. " + "x" * 140
_LONG_BLOCK_B = "Always classify the user message into exactly one category. " + "y" * 140


def _message(usage_metadata=None, response_metadata=None):
    return SimpleNamespace(
        content="hi",
        usage_metadata=usage_metadata,
        response_metadata=response_metadata or {},
    )


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_defaults_are_zero(self):
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_total_tokens_sums_input_output(self):
        assert TokenUsage(input_tokens=100, output_tokens=25).total_tokens == 125

    def test_merged_sums_fields(self):
        a = TokenUsage(input_tokens=100, output_tokens=10, cached_input_tokens=40, cache_write_tokens=5)
        b = TokenUsage(input_tokens=50, output_tokens=2, cached_input_tokens=10)
        merged = TokenUsage.merged([a, b])
        assert merged.input_tokens == 150
        assert merged.output_tokens == 12
        assert merged.cached_input_tokens == 50
        assert merged.cache_write_tokens == 5

    def test_merged_empty_returns_none(self):
        assert TokenUsage.merged([]) is None


# ---------------------------------------------------------------------------
# usage_from_message
# ---------------------------------------------------------------------------


class TestUsageFromMessage:
    def test_prefers_normalized_usage_metadata(self):
        msg = _message(
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_token_details": {"cache_read": 50, "cache_creation": 10},
            }
        )
        usage = usage_from_message(msg)
        assert usage == TokenUsage(input_tokens=100, output_tokens=20, cached_input_tokens=50, cache_write_tokens=10)

    def test_anthropic_response_metadata_shape(self):
        # Anthropic's input_tokens EXCLUDES cached tokens — the extractor adds them.
        msg = _message(
            response_metadata={
                "usage": {
                    "input_tokens": 40,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 50,
                    "cache_creation_input_tokens": 10,
                }
            }
        )
        usage = usage_from_message(msg)
        assert usage.input_tokens == 100  # 40 + 50 + 10
        assert usage.cached_input_tokens == 50
        assert usage.cache_write_tokens == 10
        assert usage.output_tokens == 5

    def test_openai_token_usage_shape(self):
        # OpenAI's prompt_tokens INCLUDES cached tokens.
        msg = _message(
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 30,
                    "prompt_tokens_details": {"cached_tokens": 60},
                }
            }
        )
        usage = usage_from_message(msg)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 30
        assert usage.cached_input_tokens == 60
        assert usage.cache_write_tokens == 0

    def test_no_usage_returns_none(self):
        assert usage_from_message(_message()) is None
        assert usage_from_message(None) is None

    def test_real_aimessage_with_usage_metadata(self):
        msg = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "input_token_details": {"cache_read": 4},
            },
        )
        usage = usage_from_message(msg)
        assert usage.input_tokens == 10
        assert usage.cached_input_tokens == 4


class TestModelNameFromMessage:
    def test_prefers_model_name(self):
        msg = _message(response_metadata={"model_name": "gpt-4o", "model": "other"})
        assert model_name_from_message(msg) == "gpt-4o"

    def test_falls_back_to_model_then_model_id(self):
        assert model_name_from_message(_message(response_metadata={"model": "claude-x"})) == "claude-x"
        assert model_name_from_message(_message(response_metadata={"model_id": "m"})) == "m"

    def test_missing_returns_empty(self):
        assert model_name_from_message(_message()) == ""


class TestEstimateTokens:
    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0

    def test_rough_four_chars_per_token(self):
        assert estimate_tokens("x" * 400) == 100


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class TestNormalizeModelName:
    def test_strips_openrouter_and_vendor(self):
        assert normalize_model_name("openrouter/anthropic/claude-3-haiku") == "claude-3-haiku"

    def test_strips_bare_vendor_prefix(self):
        assert normalize_model_name("openai/gpt-4o-mini") == "gpt-4o-mini"

    def test_local_maps_to_local(self):
        assert normalize_model_name("local/unsloth/gemma-4-12b-it-GGUF") == "local"
        assert normalize_model_name("local") == "local"

    def test_google_models_path_prefix_stripped(self):
        # Gemini's API names models "models/gemini-..." in response metadata.
        assert normalize_model_name("models/gemini-2.5-flash") == "gemini-2.5-flash"

    def test_plain_name_unchanged(self):
        assert normalize_model_name("gpt-4o") == "gpt-4o"

    def test_empty_stays_empty(self):
        assert normalize_model_name("") == ""


class TestResolvePrice:
    def test_prefix_match(self):
        assert resolve_price("claude-haiku-4-5-20251001") == (1.00, 0.10, 5.00)

    def test_longest_prefix_wins(self):
        # "gpt-4o-mini" is longer than "gpt-4o" and must win for mini names.
        assert resolve_price("gpt-4o-mini-2024-07-18") == (0.15, 0.075, 0.60)

    def test_specific_older_family_entry(self):
        assert resolve_price("claude-3-5-haiku-20241022") == (0.80, 0.08, 4.00)

    def test_gemini_family(self):
        assert resolve_price("gemini-2.5-pro") == (1.25, 0.31, 10.00)
        assert resolve_price("gemini-2.5-flash") == (0.10, 0.025, 0.40)
        assert resolve_price("models/gemini-2.0-flash") == (0.10, 0.025, 0.40)

    def test_local_is_free(self):
        assert resolve_price("local/whatever") == (0.0, 0.0, 0.0)

    def test_empty_name_uses_default(self):
        assert resolve_price("") == (2.50, 1.25, 10.00)

    def test_unknown_model_uses_default(self):
        assert resolve_price("totally-unknown-model") == (2.50, 1.25, 10.00)

    def test_openrouter_path_resolves_inner_model(self):
        assert resolve_price("openrouter/openai/gpt-4o") == (2.50, 1.25, 10.00)

    def test_overrides_beat_builtin_table(self):
        overrides = {"gpt-4o": (1.0, 0.5, 2.0)}
        assert resolve_price("gpt-4o", overrides) == (1.0, 0.5, 2.0)


class TestEstimateCost:
    def test_uncached_math(self):
        usage = TokenUsage(input_tokens=1500, output_tokens=40)
        cost = estimate_cost("gpt-4o", usage)
        assert cost == pytest.approx((1500 * 2.5 + 40 * 10) / 1e6)

    def test_cached_tokens_billed_at_discount(self):
        usage = TokenUsage(input_tokens=1500, output_tokens=40, cached_input_tokens=500)
        cost = estimate_cost("gpt-4o", usage)
        assert cost == pytest.approx((1000 * 2.5 + 500 * 1.25 + 40 * 10) / 1e6)

    def test_cache_write_premium(self):
        # Anthropic-style: 1.25x on cache-creation tokens.
        usage = TokenUsage(input_tokens=1000, output_tokens=10, cache_write_tokens=200)
        cost = estimate_cost("claude-haiku-4-5-20251001", usage)
        assert cost == pytest.approx((800 * 1.0 + 200 * 1.0 * 1.25 + 10 * 5) / 1e6)

    def test_none_usage_is_free(self):
        assert estimate_cost("gpt-4o", None) == 0.0

    def test_local_model_is_free(self):
        assert estimate_cost("local/foo", TokenUsage(input_tokens=5000, output_tokens=500)) == 0.0

    def test_overrides_flow_through(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
        assert estimate_cost("mystery", usage, {"mystery": (3.0, 0.0, 0.0)}) == pytest.approx(3.0)


class TestPriceOverridesFromEnv:
    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("CONNTRAIL_PRICE_OVERRIDES", raising=False)
        assert price_overrides_from_env() is None

    def test_valid_json(self, monkeypatch):
        monkeypatch.setenv("CONNTRAIL_PRICE_OVERRIDES", json.dumps({"gpt-9": [1, 0.5, 2]}))
        assert price_overrides_from_env() == {"gpt-9": (1.0, 0.5, 2.0)}

    def test_invalid_json_raises(self, monkeypatch):
        monkeypatch.setenv("CONNTRAIL_PRICE_OVERRIDES", "{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            price_overrides_from_env()

    def test_non_object_raises(self, monkeypatch):
        monkeypatch.setenv("CONNTRAIL_PRICE_OVERRIDES", json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="JSON object"):
            price_overrides_from_env()

    def test_wrong_arity_raises(self, monkeypatch):
        monkeypatch.setenv("CONNTRAIL_PRICE_OVERRIDES", json.dumps({"gpt-9": [1, 2]}))
        with pytest.raises(ValueError, match="gpt-9"):
            price_overrides_from_env()

    def test_config_picks_up_env_override(self, monkeypatch):
        monkeypatch.setenv("CONNTRAIL_PRICE_OVERRIDES", json.dumps({"gpt-9": [1, 0.5, 2]}))
        config = ConntrailConfig()
        assert config.model_prices == {"gpt-9": (1.0, 0.5, 2.0)}


# ---------------------------------------------------------------------------
# CostCallbackHandler
# ---------------------------------------------------------------------------


def _start_event(handler, messages, run_id=None, serialized=None):
    handler.on_chat_model_start(
        serialized or {},
        [messages],
        run_id=run_id or uuid4(),
    )


def _end_event(handler, message, run_id=None):
    generations = [[SimpleNamespace(message=message)]]
    handler.on_llm_end(SimpleNamespace(generations=generations), run_id=run_id or uuid4())


class TestCostCallbackHandler:
    def test_no_calls_summary_is_none(self):
        assert CostCallbackHandler().usage_summary() is None

    def test_records_usage_and_prompt_metrics(self):
        handler = CostCallbackHandler()
        run_id = uuid4()
        _start_event(
            handler,
            [SystemMessage(content=_LONG_BLOCK_A), HumanMessage(content="hello")],
            run_id=run_id,
        )
        _end_event(
            handler,
            _message(
                usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                response_metadata={"model_name": "gpt-4o"},
            ),
            run_id=run_id,
        )

        assert len(handler.calls) == 1
        call = handler.calls[0]
        assert call.model == "gpt-4o"
        assert call.usage.input_tokens == 100
        assert call.usage.output_tokens == 20
        assert call.message_count == 2
        assert call.system_prompt_chars == len(_LONG_BLOCK_A)
        assert call.prompt_chars == len(_LONG_BLOCK_A) + len("hello")
        assert call.prompt_hashes  # the long system block was fingerprinted

    def test_summary_aggregates_across_calls(self):
        handler = CostCallbackHandler()
        for _ in range(2):
            run_id = uuid4()
            _start_event(handler, [HumanMessage(content="hi")], run_id=run_id)
            _end_event(
                handler,
                _message(
                    usage_metadata={
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "total_tokens": 110,
                        "input_token_details": {"cache_read": 60},
                    },
                    response_metadata={"model_name": "gpt-4o"},
                ),
                run_id=run_id,
            )

        summary = handler.usage_summary()
        assert summary["llm_call_count"] == 2
        assert summary["input_tokens"] == 200
        assert summary["output_tokens"] == 20
        assert summary["cached_input_tokens"] == 120
        assert summary["total_tokens"] == 220
        assert summary["models"] == ["gpt-4o"]
        assert summary["repeated_prompt_blocks"] == 0

    def test_repeated_prompt_blocks_detected(self):
        handler = CostCallbackHandler()
        system = SystemMessage(content=_LONG_BLOCK_A + "\n\n" + _LONG_BLOCK_B)
        for _ in range(3):
            run_id = uuid4()
            _start_event(handler, [system, HumanMessage(content="q")], run_id=run_id)
            _end_event(handler, _message(), run_id=run_id)

        summary = handler.usage_summary()
        # Every call sent the same 2 blocks (+ whole-text hash) — occurrences
        # exceed uniques, so repeated_prompt_blocks > 0.
        assert summary["repeated_prompt_blocks"] > 0
        assert summary["llm_call_count"] == 3

    def test_short_prompts_are_not_fingerprinted(self):
        handler = CostCallbackHandler()
        run_id = uuid4()
        _start_event(handler, [SystemMessage(content="be brief")], run_id=run_id)
        _end_event(handler, _message(), run_id=run_id)
        assert handler.calls[0].prompt_hashes == []

    def test_content_block_lists_are_extracted(self):
        handler = CostCallbackHandler()
        run_id = uuid4()
        block_message = SimpleNamespace(
            content=[{"type": "text", "text": _LONG_BLOCK_A}],
            usage_metadata=None,
            response_metadata={},
        )
        handler.on_chat_model_start({}, [[block_message]], run_id=run_id)
        _end_event(handler, _message(), run_id=run_id)
        assert handler.calls[0].prompt_chars == len(_LONG_BLOCK_A)

    def test_end_without_start_still_records_usage(self):
        handler = CostCallbackHandler()
        _end_event(handler, _message(usage_metadata={"input_tokens": 5, "output_tokens": 1}))
        assert handler.calls[0].usage.input_tokens == 5
        assert handler.calls[0].prompt_chars == 0

    def test_estimated_cost_sums_per_call_models(self):
        handler = CostCallbackHandler()
        for model, usage in (
            ("gpt-4o", TokenUsage(input_tokens=1000, output_tokens=100)),
            ("claude-haiku-4-5-20251001", TokenUsage(input_tokens=2000, output_tokens=100)),
        ):
            run_id = uuid4()
            _start_event(handler, [HumanMessage(content="x")], run_id=run_id)
            _end_event(
                handler,
                _message(
                    usage_metadata={
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                    response_metadata={"model_name": model},
                ),
                run_id=run_id,
            )
        cost = handler.estimated_cost()
        expected = (
            (1000 * 2.5 + 100 * 10) / 1e6 + (2000 * 1.0 + 100 * 5) / 1e6
        )
        assert cost == pytest.approx(expected)

    def test_estimated_cost_none_without_usage(self):
        handler = CostCallbackHandler()
        run_id = uuid4()
        _start_event(handler, [HumanMessage(content="x")], run_id=run_id)
        _end_event(handler, _message(), run_id=run_id)
        assert handler.estimated_cost() is None


# ---------------------------------------------------------------------------
# llm_cost_capture (contextvar plumbing)
# ---------------------------------------------------------------------------


class TestLLMCostCapture:
    def test_sets_and_resets_contextvar(self):
        handler = CostCallbackHandler()
        assert var_child_runnable_config.get() is None
        with llm_cost_capture(handler):
            config = var_child_runnable_config.get()
            assert config is not None
            assert handler in config["callbacks"]
        assert var_child_runnable_config.get() is None

    def test_merges_with_existing_list_callbacks(self):
        other = BaseCallbackHandler()
        token = var_child_runnable_config.set({"callbacks": [other], "tags": ["host"]})
        try:
            handler = CostCallbackHandler()
            with llm_cost_capture(handler):
                config = var_child_runnable_config.get()
                assert other in config["callbacks"]
                assert handler in config["callbacks"]
                assert config["tags"] == ["host"]
        finally:
            var_child_runnable_config.reset(token)

    def test_copies_host_callback_manager_without_mutating_it(self):
        host_manager = AsyncCallbackManager(handlers=[BaseCallbackHandler()])
        token = var_child_runnable_config.set({"callbacks": host_manager})
        try:
            handler = CostCallbackHandler()
            with llm_cost_capture(handler):
                config = var_child_runnable_config.get()
                merged = config["callbacks"]
                assert merged is not host_manager
                assert handler in merged.handlers
                # The host's own manager was left untouched.
                assert handler not in host_manager.handlers
        finally:
            var_child_runnable_config.reset(token)


# ---------------------------------------------------------------------------
# End-to-end: a real (fake) LangChain chat model under capture
# ---------------------------------------------------------------------------


class _UsageChatModel(BaseChatModel):
    """Minimal chat model reporting fixed usage — exercises the real
    BaseChatModel callback plumbing without any network."""

    fake_model_name: str = "gpt-4o"
    in_tokens: int = 100
    out_tokens: int = 7
    cached: int = 0

    @property
    def _llm_type(self) -> str:
        return "usage-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": self.in_tokens,
                "output_tokens": self.out_tokens,
                "total_tokens": self.in_tokens + self.out_tokens,
                "input_token_details": {"cache_read": self.cached},
            },
            response_metadata={"model_name": self.fake_model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


async def test_real_chat_model_call_is_captured_end_to_end():
    model = _UsageChatModel()
    handler = CostCallbackHandler()

    async def node(state):
        await model.ainvoke([HumanMessage(content="classify this")])
        return {**state, "route": "general"}

    with llm_cost_capture(handler):
        await node({"message": "classify this"})

    summary = handler.usage_summary()
    assert summary["llm_call_count"] == 1
    assert summary["input_tokens"] == 100
    assert summary["output_tokens"] == 7
    assert summary["cached_input_tokens"] == 0
    assert summary["models"] == ["gpt-4o"]
    assert handler.estimated_cost() == pytest.approx((100 * 2.5 + 7 * 10) / 1e6)


# ---------------------------------------------------------------------------
# CostCapture + build_analysis_overhead
# ---------------------------------------------------------------------------


class TestCostCapture:
    def test_finish_records_latency_ms(self):
        capture = CostCapture()
        capture.finish(0.1234)
        assert capture.latency_ms == pytest.approx(123.4)

    def test_usage_summary_delegates_to_handler(self):
        capture = CostCapture()
        assert capture.usage_summary() is None


class TestBuildAnalysisOverhead:
    def test_combines_re_runs_and_contrast(self):
        overhead = build_analysis_overhead(
            re_run_usage=TokenUsage(input_tokens=4000, output_tokens=80),
            re_run_models=["gpt-4o"],
            re_run_latency_ms=1500.0,
            retries=2,
            contrast_usage=TokenUsage(input_tokens=600, output_tokens=90),
            contrast_model="claude-haiku-4-5-20251001",
        )
        assert overhead["input_tokens"] == 4600
        assert overhead["output_tokens"] == 170
        assert overhead["total_tokens"] == 4770
        assert overhead["retries"] == 2
        assert overhead["latency_ms"] == 1500.0
        expected = (4000 * 2.5 + 80 * 10) / 1e6 + (600 * 1.0 + 90 * 5) / 1e6
        assert overhead["cost_usd"] == pytest.approx(expected)

    def test_no_usage_still_reports_retries_and_latency(self):
        overhead = build_analysis_overhead(
            re_run_usage=None,
            re_run_models=[],
            re_run_latency_ms=99.0,
            retries=1,
            contrast_usage=None,
            contrast_model="claude-haiku-4-5-20251001",
        )
        assert overhead["total_tokens"] == 0
        assert overhead["cost_usd"] == 0.0
        assert overhead["retries"] == 1
