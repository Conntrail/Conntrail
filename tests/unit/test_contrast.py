"""
Tests for ContrastGenerator.

Unit tests use a mocked LLM — no API key required.
Integration tests (marked @pytest.mark.integration) call a real LLM.
"""
import json
import os

import pytest

from conntrail.contrast import ContrastGenerationError, ContrastGenerator, ContrastSet

# ---------------------------------------------------------------------------
# ContrastSet unit tests
# ---------------------------------------------------------------------------

class TestContrastSet:
    def test_as_list_order(self):
        cs = ContrastSet(similar="a", neutral="b", opposite="c")
        assert cs.as_list() == ["a", "b", "c"]

    def test_all_non_empty_true(self):
        cs = ContrastSet(
            similar="this is a similar text",
            neutral="this is a neutral text",
            opposite="this is an opposite text",
        )
        assert cs.all_non_empty() is True

    def test_all_non_empty_false_on_short(self):
        cs = ContrastSet(similar="hi", neutral="ok", opposite="no")
        assert cs.all_non_empty() is False


# ---------------------------------------------------------------------------
# _parse_response unit tests (no LLM needed)
# ---------------------------------------------------------------------------

class TestParseResponse:
    def setup_method(self):
        self.gen = ContrastGenerator(llm=object())  # dummy llm — not called here

    def _valid_json(self, similar="A similar sentence here.", neutral="A neutral sentence here.", opposite="An opposite sentence here."):
        return json.dumps({"similar": similar, "neutral": neutral, "opposite": opposite})

    def test_valid_json(self):
        cs = self.gen._parse_response(self._valid_json())
        assert cs.similar == "A similar sentence here."
        assert cs.neutral == "A neutral sentence here."
        assert cs.opposite == "An opposite sentence here."

    def test_strips_whitespace_from_values(self):
        raw = json.dumps({"similar": "  trimmed  ", "neutral": "  me  ", "opposite": "  too  "})
        cs = self.gen._parse_response(raw)
        assert cs.similar == "trimmed"
        assert cs.neutral == "me"
        assert cs.opposite == "too"

    def test_strips_markdown_code_fence(self):
        raw = '```json\n' + self._valid_json() + '\n```'
        cs = self.gen._parse_response(raw)
        assert isinstance(cs, ContrastSet)

    def test_strips_plain_code_fence(self):
        raw = '```\n' + self._valid_json() + '\n```'
        cs = self.gen._parse_response(raw)
        assert isinstance(cs, ContrastSet)

    def test_extracts_json_from_surrounding_prose(self):
        raw = f"Here is the output:\n{self._valid_json()}\nThat's all."
        cs = self.gen._parse_response(raw)
        assert isinstance(cs, ContrastSet)

    def test_recovers_truncated_json(self):
        # Simulates a response cut off by max_tokens before the closing brace.
        raw = '{"similar": "A similar sentence here.", "neutral": "A neutral sentence here.", "opposite": "An opposite sentence here.'
        with pytest.raises(ContrastGenerationError):
            self.gen._parse_response(raw)

    def test_extracts_nested_dict_value(self):
        raw = json.dumps({
            "similar": {"input": "A similar sentence here."},
            "neutral": "A neutral sentence here.",
            "opposite": "An opposite sentence here.",
        })
        cs = self.gen._parse_response(raw)
        assert cs.similar == "A similar sentence here."

    def test_raises_on_no_json(self):
        with pytest.raises(ContrastGenerationError, match="No JSON object found"):
            self.gen._parse_response("I cannot generate contrasts for this input.")

    def test_raises_on_malformed_json(self):
        with pytest.raises(ContrastGenerationError, match="Invalid JSON"):
            self.gen._parse_response('{"similar": "ok", "neutral": broken}')

    def test_raises_on_missing_key(self):
        raw = json.dumps({"similar": "ok", "neutral": "ok"})  # missing 'opposite'
        with pytest.raises(ContrastGenerationError, match="Missing required key"):
            self.gen._parse_response(raw)

    def test_raises_on_empty_value(self):
        raw = json.dumps({"similar": "", "neutral": "ok", "opposite": "ok"})
        with pytest.raises(ContrastGenerationError, match="Empty or invalid value"):
            self.gen._parse_response(raw)

    def test_raises_on_non_string_value(self):
        raw = json.dumps({"similar": 123, "neutral": "ok", "opposite": "ok"})
        with pytest.raises(ContrastGenerationError, match="Empty or invalid value"):
            self.gen._parse_response(raw)

    def test_raises_on_unextractable_nested_dict(self):
        raw = json.dumps({
            "similar": {"unrelated_key": "value"},
            "neutral": "ok value here",
            "opposite": "ok value here",
        })
        with pytest.raises(ContrastGenerationError, match="Could not extract string"):
            self.gen._parse_response(raw)


# ---------------------------------------------------------------------------
# ContrastGenerator unit tests with mocked LLM
# ---------------------------------------------------------------------------

VALID_RESPONSE = json.dumps({
    "similar": "I need this resolved as soon as possible, it's critical.",
    "neutral": "Please look into this issue.",
    "opposite": "There is no rush on this, handle it when convenient.",
})


class TestContrastGeneratorMocked:
    def _make_gen(self, mocker, response_content=VALID_RESPONSE):
        mock_llm = mocker.MagicMock()
        mock_llm.ainvoke = mocker.AsyncMock(
            return_value=mocker.MagicMock(content=response_content)
        )
        return ContrastGenerator(llm=mock_llm)

    def test_instantiation_default_model(self):
        gen = ContrastGenerator()
        assert gen.model == "claude-haiku-4-5-20251001"

    def test_instantiation_custom_model(self):
        gen = ContrastGenerator(model="llama-3.1-8b-instant")
        assert gen.model == "llama-3.1-8b-instant"

    def test_llm_kwarg_skips_auto_build(self, mocker):
        sentinel = object()
        gen = ContrastGenerator(llm=sentinel)
        assert gen._llm is sentinel

    @pytest.mark.asyncio
    async def test_generate_returns_contrast_set(self, mocker):
        gen = self._make_gen(mocker)
        result = await gen.generate("I need this fixed ASAP, system is down")
        assert isinstance(result, ContrastSet)
        assert result.all_non_empty()

    @pytest.mark.asyncio
    async def test_generate_calls_llm_once(self, mocker):
        gen = self._make_gen(mocker)
        await gen.generate("test input")
        gen._llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_prompt_contains_input(self, mocker):
        gen = self._make_gen(mocker)
        input_text = "my specific input string"
        await gen.generate(input_text)
        call_args = gen._llm.ainvoke.call_args[0][0]  # first positional arg (messages list)
        assert any(input_text in msg.content for msg in call_args)

    @pytest.mark.asyncio
    async def test_generate_empty_input_raises(self, mocker):
        gen = self._make_gen(mocker)
        with pytest.raises(ContrastGenerationError, match="cannot be empty"):
            await gen.generate("")

    @pytest.mark.asyncio
    async def test_generate_whitespace_only_raises(self, mocker):
        gen = self._make_gen(mocker)
        with pytest.raises(ContrastGenerationError, match="cannot be empty"):
            await gen.generate("   ")

    @pytest.mark.asyncio
    async def test_generate_propagates_parse_error(self, mocker):
        gen = self._make_gen(mocker, response_content="not valid json at all")
        with pytest.raises(ContrastGenerationError):
            await gen.generate("some input")

    @pytest.mark.asyncio
    async def test_generate_llm_failure_raises(self, mocker):
        mock_llm = mocker.MagicMock()
        mock_llm.ainvoke = mocker.AsyncMock(side_effect=Exception("API timeout"))
        gen = ContrastGenerator(llm=mock_llm)
        with pytest.raises(ContrastGenerationError, match="LLM call failed"):
            await gen.generate("test")


# ---------------------------------------------------------------------------
# Integration tests — real LLM, real API key
# ---------------------------------------------------------------------------

FIXTURE_INPUTS = [
    ("I need this fixed ASAP, system is down",           "urgency"),
    ("Can you explain how embeddings work?",              "technical neutral"),
    ("I'm not sure if I should upgrade or stay",         "ambiguous"),
    ("This is absolutely wonderful, couldn't be happier","emotional positive"),
    ("x",                                                 "minimal input"),
]


@pytest.mark.integration
class TestContrastGeneratorIntegration:

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not any(os.getenv(k) for k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
            pytest.skip("No API key available")

    def _make_gen(self):
        from conntrail.utils.providers import _FALLBACK_MODELS, _available_provider
        provider = _available_provider()
        model = _FALLBACK_MODELS[provider]
        return ContrastGenerator(model=model)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text,label", FIXTURE_INPUTS)
    async def test_generate_fixture(self, input_text, label):
        gen = self._make_gen()
        result = await gen.generate(input_text)
        assert isinstance(result, ContrastSet), f"[{label}] Expected ContrastSet"
        assert result.all_non_empty(), f"[{label}] All variants must be >= 5 chars: {result}"

    @pytest.mark.asyncio
    async def test_variants_differ_from_each_other(self):
        gen = self._make_gen()
        result = await gen.generate("I need a refund immediately, this is unacceptable!")
        variants = result.as_list()
        assert len(set(variants)) == 3, f"All 3 variants must be distinct: {variants}"

    @pytest.mark.asyncio
    async def test_opposite_inverts_dimension(self):
        gen = self._make_gen()
        result = await gen.generate("I need this escalated immediately, system is critical!")
        assert result.opposite.strip(), "Opposite must not be empty"
        assert result.opposite != result.similar, (
            "Opposite should differ from similar variant"
        )
