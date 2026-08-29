"""
Tests for DivergenceAnalyser.

All tests use the deterministic mock_router_node from fixtures — no API key needed.
mock_router_node routes:  urgency keywords → "escalate",  everything else → "general"
"""
import asyncio
import time

import pytest

from conntrail.analyser import AnalysisResult, DivergenceAnalyser, RetryExhaustedError
from conntrail.contrast import ContrastSet
from tests.fixtures.sample_graphs import mock_router_node

# ---------------------------------------------------------------------------
# DivergenceAnalyser._extract_route
# ---------------------------------------------------------------------------

class TestExtractRoute:
    def setup_method(self):
        self.analyser = DivergenceAnalyser()

    def test_explicit_route_key(self):
        inp = {"message": "hi", "category": None}
        out = {"message": "hi", "category": "refund"}
        assert self.analyser._extract_route(inp, out, route_key="category") == "refund"

    def test_explicit_route_key_missing_returns_unknown(self):
        out = {"message": "hi"}
        assert self.analyser._extract_route({}, out, route_key="missing_key") == "unknown"

    def test_auto_detect_none_to_string(self):
        inp = {"message": "hi", "category": None}
        out = {"message": "hi", "category": "escalation"}
        assert self.analyser._extract_route(inp, out) == "escalation"

    def test_auto_detect_changed_string(self):
        inp = {"message": "hi", "strategy": "old"}
        out = {"message": "hi", "strategy": "vector_search"}
        assert self.analyser._extract_route(inp, out) == "vector_search"

    def test_auto_detect_prefers_none_to_string_over_changed(self):
        # Both None→string and changed-string present — None→string wins (comes first)
        inp = {"message": "hi", "category": None, "strategy": "old"}
        out = {"message": "hi", "category": "refund", "strategy": "new"}
        result = self.analyser._extract_route(inp, out)
        assert result in ("refund", "new")  # either is acceptable

    def test_auto_detect_no_change_returns_unknown(self):
        inp = {"message": "hi"}
        out = {"message": "hi"}
        assert self.analyser._extract_route(inp, out) == "unknown"


# ---------------------------------------------------------------------------
# DivergenceAnalyser._infer_attribution
# ---------------------------------------------------------------------------

class TestInferAttribution:
    def setup_method(self):
        self.analyser = DivergenceAnalyser()

    def test_opposite_flips_names_semantic_intensity(self):
        attr, cf = self.analyser._infer_attribution(
            original_route="escalate",
            contrast_routes={"similar": "escalate", "neutral": "escalate", "opposite": "general"},
        )
        assert attr == "semantic intensity"
        assert cf == "general"

    def test_neutral_flips_names_urgency_sentiment(self):
        # Only neutral flips — opposite stays same, so neutral wins
        attr, cf = self.analyser._infer_attribution(
            original_route="escalate",
            contrast_routes={"similar": "escalate", "neutral": "general", "opposite": "escalate"},
        )
        assert attr == "urgency/sentiment"
        assert cf == "general"

    def test_similar_flips_names_surface_form(self):
        # Only similar flips — neutral and opposite stay same
        attr, cf = self.analyser._infer_attribution(
            original_route="escalate",
            contrast_routes={"similar": "general", "neutral": "escalate", "opposite": "escalate"},
        )
        assert attr == "surface form"
        assert cf == "general"

    def test_no_flip_returns_none_detected(self):
        attr, cf = self.analyser._infer_attribution(
            original_route="escalate",
            contrast_routes={"similar": "escalate", "neutral": "escalate", "opposite": "escalate"},
        )
        assert attr == "none detected"
        assert cf is None

    def test_priority_opposite_over_neutral(self):
        # Both neutral and opposite flip — opposite should win
        attr, cf = self.analyser._infer_attribution(
            original_route="A",
            contrast_routes={"similar": "A", "neutral": "B", "opposite": "C"},
        )
        assert attr == "semantic intensity"
        assert cf == "C"


# ---------------------------------------------------------------------------
# DivergenceAnalyser.analyse — with deterministic mock_router_node
# ---------------------------------------------------------------------------

class TestDivergenceAnalyser:
    def setup_method(self):
        self.analyser = DivergenceAnalyser()

    @pytest.mark.asyncio
    async def test_urgent_input_high_entropy(self):
        """Urgent input + contrasts that flip the route → entropy > 0.5."""
        # similar keeps urgency → escalate, neutral + opposite drop it → general
        contrasts = ContrastSet(
            similar="This needs immediate attention, it's critical!",   # → escalate
            neutral="Please look into this issue when you can.",         # → general
            opposite="There is no rush on this, handle when convenient.",# → general
        )
        original = {"message": "I need this fixed ASAP!", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert result.entropy_score >= 0.5, f"Expected high entropy, got {result.entropy_score}"

    @pytest.mark.asyncio
    async def test_routine_input_low_entropy(self):
        """Routine input — only opposite flips → low entropy."""
        contrasts = ContrastSet(
            similar="When do you open?",                                 # → general
            neutral="What are your business hours?",                    # → general
            opposite="I need to know your hours IMMEDIATELY, urgent!",  # → escalate
        )
        original = {"message": "What are your business hours?", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert result.entropy_score < 0.8, f"Expected lower entropy, got {result.entropy_score}"
        assert result.original_route == "general"

    @pytest.mark.asyncio
    async def test_confident_input_zero_entropy(self):
        """All 4 variants route the same way → entropy = 0.0."""
        contrasts = ContrastSet(
            similar="This is absolutely urgent and critical!",
            neutral="Please address this critical system issue.",
            opposite="This is an emergency that needs immediate action!",
        )
        original = {"message": "URGENT: system is down!", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert result.entropy_score == 0.0
        assert result.original_route == "escalate"
        assert result.attribution_dimension == "none detected"
        assert result.counterfactual_route is None

    @pytest.mark.asyncio
    async def test_attribution_dimension_detected(self):
        """Attribution dimension is a non-empty string when route flips."""
        contrasts = ContrastSet(
            similar="I need this urgently!",
            neutral="Please help with this.",
            opposite="No rush on this at all.",
        )
        original = {"message": "Fix this ASAP!", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert result.attribution_dimension
        assert len(result.attribution_dimension) > 2

    @pytest.mark.asyncio
    async def test_counterfactual_route_is_opposite_of_original(self):
        """When route flips, counterfactual_route is the flipped destination."""
        contrasts = ContrastSet(
            similar="Fix this immediately!",     # → escalate
            neutral="Please fix this issue.",    # → general (flips)
            opposite="No rush, low priority.",   # → general (flips)
        )
        original = {"message": "Fix this ASAP!", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert result.original_route == "escalate"
        assert result.counterfactual_route == "general"

    @pytest.mark.asyncio
    async def test_raw_outputs_contains_all_variants(self):
        """raw_outputs must contain keys for all 4 variants."""
        contrasts = ContrastSet(similar="s", neutral="n", opposite="o")
        original = {"message": "hello", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert set(result.raw_outputs.keys()) == {"original", "similar", "neutral", "opposite"}

    @pytest.mark.asyncio
    async def test_contrast_routes_contains_three_keys(self):
        contrasts = ContrastSet(similar="s", neutral="n", opposite="o")
        original = {"message": "hello", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert set(result.contrast_routes.keys()) == {"similar", "neutral", "opposite"}

    @pytest.mark.asyncio
    async def test_explicit_route_key(self):
        """Explicit route_key bypasses auto-detection."""
        contrasts = ContrastSet(
            similar="urgent fix needed asap",
            neutral="please help",
            opposite="no rush",
        )
        original = {"message": "Fix this ASAP!", "route": None, "response": None}
        result = await self.analyser.analyse(
            mock_router_node, original, contrasts, route_key="route"
        )
        assert result.original_route == "escalate"

    @pytest.mark.asyncio
    async def test_sync_node_supported(self):
        """analyse() works with a synchronous node function."""
        def sync_router(state):
            text = state["message"].lower()
            route = "escalate" if "urgent" in text else "general"
            return {**state, "route": route}

        contrasts = ContrastSet(similar="urgent!", neutral="please help", opposite="no rush")
        original = {"message": "urgent!", "route": None, "response": None}
        result = await self.analyser.analyse(sync_router, original, contrasts, route_key="route")
        assert result.original_route == "escalate"

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """All 4 node calls run concurrently — total time ≈ 1 call, not 4."""
        async def slow_node(state):
            await asyncio.sleep(0.1)
            return {**state, "route": "general"}

        contrasts = ContrastSet(similar="s", neutral="n", opposite="o")
        original = {"message": "hi", "route": None, "response": None}

        start = time.perf_counter()
        await self.analyser.analyse(slow_node, original, contrasts, route_key="route")
        elapsed = time.perf_counter() - start

        # 4 concurrent 0.1s sleeps should complete in ~0.1s, not ~0.4s
        assert elapsed < 0.3, f"Expected concurrent execution (~0.1s), took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_analysis_result_is_dataclass(self):
        contrasts = ContrastSet(similar="s", neutral="n", opposite="o")
        original = {"message": "test", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert isinstance(result, AnalysisResult)
        assert isinstance(result.entropy_score, float)
        assert 0.0 <= result.entropy_score <= 1.0

    @pytest.mark.asyncio
    async def test_unknown_route_surfaces_when_no_route_signal(self):
        """A node that never changes any key → 'unknown' route for every variant."""
        async def no_op_node(state):
            return dict(state)

        contrasts = ContrastSet(similar="s", neutral="n", opposite="o")
        original = {"message": "hello"}
        result = await self.analyser.analyse(no_op_node, original, contrasts)
        assert result.original_route == "unknown"
        assert result.contrast_routes == {"similar": "unknown", "neutral": "unknown", "opposite": "unknown"}
        assert result.entropy_score == 0.0
        assert result.attribution_dimension == "none detected"


# ---------------------------------------------------------------------------
# DivergenceAnalyser._parse_retry_after
# ---------------------------------------------------------------------------

class TestParseRetryAfter:
    def setup_method(self):
        self.analyser = DivergenceAnalyser()

    def test_minutes_and_seconds(self):
        delay = self.analyser._parse_retry_after("Please try again in 1m50.592s")
        assert delay == pytest.approx(1 * 60 + 50.592 + 2.0)

    def test_seconds_only(self):
        delay = self.analyser._parse_retry_after("try again in 5s")
        assert delay == pytest.approx(5.0 + 2.0)

    def test_no_match_returns_none(self):
        assert self.analyser._parse_retry_after("some unrelated error message") is None


# ---------------------------------------------------------------------------
# DivergenceAnalyser._call_node — rate-limit retry + exhaustion
# ---------------------------------------------------------------------------

class TestCallNodeRetry:
    def setup_method(self):
        self.analyser = DivergenceAnalyser()

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(self, mocker):
        mocker.patch("conntrail.analyser.asyncio.sleep", new=mocker.AsyncMock())
        calls = {"count": 0}

        async def flaky_node(state):
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError("429 rate_limit exceeded, try again in 1s")
            return {**state, "route": "general"}

        result = await self.analyser._call_node(flaky_node, {"message": "hi"})
        assert result["route"] == "general"
        assert calls["count"] == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries_raises_retry_exhausted_error(self, mocker):
        mocker.patch("conntrail.analyser.asyncio.sleep", new=mocker.AsyncMock())
        calls = {"count": 0}

        async def always_rate_limited(state):
            calls["count"] += 1
            raise RuntimeError("429 rate limit")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await self.analyser._call_node(always_rate_limited, {"message": "hi"})
        assert calls["count"] == 4  # max_retries
        assert isinstance(exc_info.value.original_error, RuntimeError)

    @pytest.mark.asyncio
    async def test_non_rate_limit_error_raises_immediately(self, mocker):
        sleep_spy = mocker.patch("conntrail.analyser.asyncio.sleep", new=mocker.AsyncMock())
        calls = {"count": 0}

        async def broken_node(state):
            calls["count"] += 1
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await self.analyser._call_node(broken_node, {"message": "hi"})
        assert calls["count"] == 1  # no retries attempted
        sleep_spy.assert_not_called()
