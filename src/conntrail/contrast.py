"""
ContrastGenerator — generates semantically targeted contrast variants for a given input.

Produces 3 variants via a single structured LLM call:
  - similar:  same intent, different surface form
  - neutral:  urgency/sentiment stripped, core intent preserved
  - opposite: semantic inversion of the key dimension
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

_PROMPT_PATH = Path(__file__).parent / "prompts" / "contrast_gen.txt"


class ContrastGenerationError(Exception):
    """Raised when the LLM returns an unparseable or invalid contrast response."""


@dataclass
class ContrastSet:
    """Three semantically targeted contrast variants for a given input."""

    similar: str
    neutral: str
    opposite: str

    def as_list(self) -> list[str]:
        """Return variants as an ordered list: [similar, neutral, opposite]."""
        return [self.similar, self.neutral, self.opposite]

    def all_non_empty(self) -> bool:
        return all(len(v.strip()) >= 5 for v in self.as_list())


class ContrastGenerator:
    """
    Generates 3 contrast variants for any text input using a cheap LLM.

    Uses a single structured JSON call — all 3 variants in one response.
    The model used here is intentionally separate from the model being traced.

    Args:
        model: LLM model ID string. Provider auto-detected from the name prefix,
               with fallback to whichever API key is available.
        llm:   Pass a pre-configured LangChain BaseChatModel to skip auto-detection.
               Takes precedence over `model` when provided.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.model = model
        self._llm = llm  # pre-configured model, or None to lazy-build from model string

    async def generate(self, input_text: str) -> ContrastSet:
        """
        Generate 3 contrast variants for the given input text.

        Args:
            input_text: The original agent input to contrast.

        Returns:
            ContrastSet with similar, neutral, and opposite variants.

        Raises:
            ContrastGenerationError: If input is empty or the LLM response
                                     cannot be parsed into a valid ContrastSet.
        """
        if not input_text or not input_text.strip():
            raise ContrastGenerationError("Input text cannot be empty")

        prompt = self._load_prompt_template().format(input=input_text)
        llm = self._get_llm()

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
        except Exception as e:
            raise ContrastGenerationError(f"LLM call failed: {e}") from e

        return self._parse_response(response.content)

    def _get_llm(self) -> BaseChatModel:
        """Lazy-build and cache the LangChain model instance."""
        if self._llm is None:
            from conntrail.utils.providers import get_chat_model
            self._llm = get_chat_model(self.model, max_tokens=300)
        return self._llm

    def _load_prompt_template(self) -> str:
        """Load the versioned prompt template from prompts/contrast_gen.txt."""
        try:
            return _PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ContrastGenerationError(
                f"Prompt template not found at {_PROMPT_PATH}"
            )

    def _parse_response(self, raw: str) -> ContrastSet:
        """
        Parse the LLM JSON response into a ContrastSet.

        Handles markdown code fences and extracts the JSON object even
        if the LLM adds surrounding text.

        Raises:
            ContrastGenerationError: If JSON is malformed or keys are missing/empty.
        """
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first line (```json or ```) and last line (```)
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()

        # Extract the JSON object (handles leading/trailing prose)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1:
            raise ContrastGenerationError(
                f"No JSON object found in LLM response: {raw!r}"
            )

        # Handle truncated responses (max_tokens hit) — try closing the object
        candidate = text[start:end] if end > 0 else text[start:] + "\n}"

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ContrastGenerationError(
                f"Invalid JSON in LLM response: {e}\nRaw: {raw!r}"
            ) from e

        # Validate all required keys
        extracted: dict[str, str] = {}
        for key in ("similar", "neutral", "opposite"):
            if key not in data:
                raise ContrastGenerationError(
                    f"Missing required key {key!r} in response: {data}"
                )
            value = data[key]
            # Handle nested dicts: LLMs sometimes return {key: {input: "...", ...}}
            if isinstance(value, dict):
                for nested_key in ("input", "text", "content", "value", "variant"):
                    if nested_key in value and isinstance(value[nested_key], str):
                        value = value[nested_key]
                        break
                else:
                    raise ContrastGenerationError(
                        f"Could not extract string from nested dict for {key!r}: {data[key]!r}"
                    )
            if not isinstance(value, str) or not value.strip():
                raise ContrastGenerationError(
                    f"Empty or invalid value for {key!r}: {value!r}"
                )
            extracted[key] = value.strip()

        return ContrastSet(
            similar=extracted["similar"],
            neutral=extracted["neutral"],
            opposite=extracted["opposite"],
        )
