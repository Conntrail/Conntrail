"""
Tests for ConntrailConfig and NodeInterceptor.
"""
import asyncio
import time
from datetime import UTC

import pytest

from conntrail.analyser import RetryExhaustedError
from conntrail.config import ConntrailConfig
from conntrail.exporters.base import BaseExporter
from conntrail.interceptor import NodeInterceptor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RecordingExporter(BaseExporter):
    """Test double — records every write() call instead of hitting the network."""

    def __init__(self) -> None:
        self.records = []

    async def write(self, record) -> None:
        self.records.append(record)


async def _noop_node(state):
    return state


async def _routing_node(state):
    msg = state.get("message", "")
    route = "urgent" if "urgent" in msg.lower() else "general"
    return {**state, "route": route}


def _make_interceptor(node_fn=None, input_key="message", route_key="route", **config_kwargs):
    config = ConntrailConfig(**config_kwargs)
    return NodeInterceptor(
        node_fn or _noop_node,
        node_id="test_node",
        config=config,
        input_key=input_key,
        route_key=route_key,
    )


# ---------------------------------------------------------------------------
# ConntrailConfig validation (live)
# ---------------------------------------------------------------------------

class TestConntrailConfig:
    def test_default_values(self):
        config = ConntrailConfig()
        assert config.contrast_model == "claude-haiku-4-5-20251001"
        assert config.sample_rate == 1.0
        assert config.async_mode is True
        assert config.exporter is None
        assert config.entropy_alert_threshold == 0.6

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError, match="sample_rate"):
            ConntrailConfig(sample_rate=1.5)

    def test_zero_sample_rate_valid(self):
        config = ConntrailConfig(sample_rate=0.0)
        assert config.sample_rate == 0.0

    def test_invalid_alert_threshold_raises(self):
        with pytest.raises(ValueError, match="entropy_alert_threshold"):
            ConntrailConfig(entropy_alert_threshold=1.5)

    def test_exporter_can_be_injected_directly(self):
        exporter = RecordingExporter()
        config = ConntrailConfig(exporter=exporter)
        assert config.exporter is exporter


# ---------------------------------------------------------------------------
# NodeInterceptor — core behaviour (success path)
# ---------------------------------------------------------------------------

class TestNodeInterceptor:
    def test_instantiation(self):
        interceptor = _make_interceptor()
        assert interceptor.node_id == "test_node"
        assert interceptor.__name__ == "test_node"

    @pytest.mark.asyncio
    async def test_output_unchanged(self):
        """Wrapped node return value must be identical to unwrapped."""
        state = {"message": "hello", "route": None}
        output = await _routing_node(state)

        interceptor = _make_interceptor(_routing_node, sample_rate=0.0)
        wrapped_output = await interceptor(state)

        assert wrapped_output == output

    @pytest.mark.asyncio
    async def test_output_unchanged_for_noop(self):
        state = {"message": "test", "value": 42}
        interceptor = _make_interceptor(sample_rate=0.0)
        result = await interceptor(state)
        assert result == state

    @pytest.mark.asyncio
    async def test_zero_sample_rate_fires_no_analysis(self):
        """sample_rate=0.0 must not schedule any contrast analysis tasks."""
        async def spy_node(state):
            return state

        interceptor = _make_interceptor(spy_node, sample_rate=0.0)

        tasks_before = len(asyncio.all_tasks())
        await interceptor({"message": "hello"})
        tasks_after = len(asyncio.all_tasks())

        # No new tasks should have been created
        assert tasks_after == tasks_before

    @pytest.mark.asyncio
    async def test_hot_path_not_blocked(self):
        """Wrapped node wall-time must be within 20ms of unwrapped node."""
        RUNS = 10
        TOLERANCE_MS = 20

        async def slow_analysis_node(state):
            await asyncio.sleep(0.001)  # 1ms simulated work
            return state

        interceptor = _make_interceptor(slow_analysis_node, sample_rate=0.0)

        # Measure unwrapped
        t0 = time.perf_counter()
        for _ in range(RUNS):
            await slow_analysis_node({"message": "x"})
        unwrapped_ms = (time.perf_counter() - t0) * 1000 / RUNS

        # Measure wrapped (analysis disabled so it's pure overhead)
        t0 = time.perf_counter()
        for _ in range(RUNS):
            await interceptor({"message": "x"})
        wrapped_ms = (time.perf_counter() - t0) * 1000 / RUNS

        overhead_ms = wrapped_ms - unwrapped_ms
        assert overhead_ms < TOLERANCE_MS, (
            f"Interceptor overhead {overhead_ms:.1f}ms exceeds {TOLERANCE_MS}ms tolerance"
        )

    @pytest.mark.asyncio
    async def test_sync_node_supported(self):
        """NodeInterceptor must also wrap synchronous node functions."""
        def sync_node(state):
            return {**state, "route": "sync_route"}

        interceptor = NodeInterceptor(
            sync_node, node_id="sync", config=ConntrailConfig(sample_rate=0.0)
        )
        result = await interceptor({"message": "hi"})
        assert result["route"] == "sync_route"

    def test_extract_input_text_uses_input_key(self):
        interceptor = _make_interceptor(input_key="message")
        state = {"message": "hello", "other": "world"}
        text, key = interceptor._extract_input_text(state)
        assert text == "hello"
        assert key == "message"

    def test_extract_input_text_fallback(self):
        """Falls back to first non-empty string if input_key missing, returns actual key."""
        interceptor = NodeInterceptor(
            _noop_node, node_id="n", config=ConntrailConfig(), input_key="missing_key"
        )
        state = {"content": "fallback text", "number": 42}
        text, key = interceptor._extract_input_text(state)
        assert text == "fallback text"
        assert key == "content"

    def test_extract_input_text_empty_state(self):
        interceptor = _make_interceptor()
        text, key = interceptor._extract_input_text({})
        assert text == ""

    @pytest.mark.asyncio
    async def test_export_skips_when_no_exporter_configured(self, caplog):
        """No exporter set → analysis runs but export is skipped (debug log), no error."""
        interceptor = _make_interceptor()  # exporter defaults to None
        from datetime import datetime

        from conntrail.contrast import ContrastSet
        from conntrail.record import TraceRecord

        record = TraceRecord(
            trace_id="t1",
            node_id="test_node",
            timestamp=datetime.now(UTC),
            original_input="x",
            original_route="a",
            entropy_score=0.1,
            stability="confident",
            attribution_dimension="none detected",
            plain_language_summary="test",
            raw_contrasts=ContrastSet(similar="s", neutral="n", opposite="o"),
            raw_outputs={},
        )
        with caplog.at_level("DEBUG", logger="conntrail"):
            await interceptor._export(record)  # must not raise
        assert "skipping export" in caplog.text

    @pytest.mark.asyncio
    async def test_export_calls_configured_exporter(self):
        exporter = RecordingExporter()
        interceptor = _make_interceptor(exporter=exporter)
        from datetime import datetime

        from conntrail.contrast import ContrastSet
        from conntrail.record import TraceRecord

        record = TraceRecord(
            trace_id="t1",
            node_id="test_node",
            timestamp=datetime.now(UTC),
            original_input="x",
            original_route="a",
            entropy_score=0.1,
            stability="confident",
            attribution_dimension="none detected",
            plain_language_summary="test",
            raw_contrasts=ContrastSet(similar="s", neutral="n", opposite="o"),
            raw_outputs={},
        )
        await interceptor._export(record)
        assert exporter.records == [record]

    @pytest.mark.asyncio
    async def test_on_alert_fires_above_threshold(self):
        """on_alert callback invoked when entropy >= entropy_alert_threshold."""
        alerts = []
        exporter = RecordingExporter()

        config = ConntrailConfig(
            sample_rate=1.0,
            entropy_alert_threshold=0.0,  # always fires
            exporter=exporter,
            on_alert=lambda rec: alerts.append(rec),
        )

        # Stub out the analysis to return a known high-entropy result
        async def _fast_analysis(input_state, original_output):
            from datetime import datetime

            from conntrail.contrast import ContrastSet
            from conntrail.record import TraceRecord
            record = TraceRecord(
                trace_id="t1",
                node_id="test_node",
                timestamp=datetime.now(UTC),
                original_input="x",
                original_route="a",
                entropy_score=0.75,
                stability="fragile",
                attribution_dimension="semantic intensity",
                plain_language_summary="test",
                raw_contrasts=ContrastSet(similar="s", neutral="n", opposite="o"),
                raw_outputs={"original": "a", "similar": "b", "neutral": "c", "opposite": "d"},
            )
            await interceptor._export(record)
            if config.on_alert and record.entropy_score >= config.entropy_alert_threshold:
                config.on_alert(record)

        interceptor = NodeInterceptor(_noop_node, node_id="test_node", config=config)
        interceptor._run_contrast_analysis = _fast_analysis

        await interceptor({"message": "trigger"})
        # Give the task time to execute
        await asyncio.sleep(0.05)
        assert len(alerts) == 1
        assert alerts[0].entropy_score == 0.75
        assert len(exporter.records) == 1


# ---------------------------------------------------------------------------
# NodeInterceptor — exception handling (the F6 fix)
# ---------------------------------------------------------------------------

class TestNodeExceptionHandling:
    @pytest.mark.asyncio
    async def test_node_exception_is_recorded_and_reraised(self):
        exporter = RecordingExporter()

        async def broken_node(state):
            raise ValueError("boom")

        interceptor = NodeInterceptor(
            broken_node, node_id="broken", config=ConntrailConfig(exporter=exporter, sample_rate=0.0)
        )

        with pytest.raises(ValueError, match="boom"):
            await interceptor({"message": "hi"})

        assert len(exporter.records) == 1
        record = exporter.records[0]
        assert record.status == "error"
        assert record.error_type == "ValueError"
        assert record.error_message == "boom"
        assert record.stability == "fragile"
        assert record.entropy_score == 1.0

    @pytest.mark.asyncio
    async def test_sync_node_exception_is_recorded_and_reraised(self):
        exporter = RecordingExporter()

        def broken_sync_node(state):
            raise KeyError("missing")

        interceptor = NodeInterceptor(
            broken_sync_node, node_id="broken_sync",
            config=ConntrailConfig(exporter=exporter, sample_rate=0.0),
        )

        with pytest.raises(KeyError):
            await interceptor({"message": "hi"})

        assert exporter.records[0].error_type == "KeyError"

    @pytest.mark.asyncio
    async def test_node_exception_skips_contrast_analysis(self, mocker):
        """No ContrastGenerator/LLM calls happen on the error path."""
        exporter = RecordingExporter()
        analyse_spy = mocker.patch("conntrail.analyser.DivergenceAnalyser.analyse")
        generate_spy = mocker.patch("conntrail.contrast.ContrastGenerator.generate")

        async def broken_node(state):
            raise RuntimeError("nope")

        interceptor = NodeInterceptor(
            broken_node, node_id="broken", config=ConntrailConfig(exporter=exporter, sample_rate=1.0)
        )

        with pytest.raises(RuntimeError):
            await interceptor({"message": "hi"})

        analyse_spy.assert_not_called()
        generate_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_exhausted_classified_distinctly(self):
        exporter = RecordingExporter()

        async def broken_node(state):
            raise RetryExhaustedError(RuntimeError("429 too many requests"))

        interceptor = NodeInterceptor(
            broken_node, node_id="broken", config=ConntrailConfig(exporter=exporter, sample_rate=0.0)
        )

        with pytest.raises(RetryExhaustedError):
            await interceptor({"message": "hi"})

        record = exporter.records[0]
        assert record.error_type == "retry_loop"

    @pytest.mark.asyncio
    async def test_node_exception_fires_on_alert(self):
        alerts = []
        exporter = RecordingExporter()

        async def broken_node(state):
            raise ValueError("boom")

        interceptor = NodeInterceptor(
            broken_node,
            node_id="broken",
            config=ConntrailConfig(
                exporter=exporter,
                sample_rate=0.0,
                entropy_alert_threshold=0.9,
                on_alert=lambda rec: alerts.append(rec),
            ),
        )

        with pytest.raises(ValueError):
            await interceptor({"message": "hi"})

        assert len(alerts) == 1
        assert alerts[0].status == "error"

    @pytest.mark.asyncio
    async def test_node_exception_exported_regardless_of_sample_rate(self):
        """Failures are always recorded — not gated by sample_rate."""
        exporter = RecordingExporter()

        async def broken_node(state):
            raise ValueError("boom")

        interceptor = NodeInterceptor(
            broken_node, node_id="broken", config=ConntrailConfig(exporter=exporter, sample_rate=0.0)
        )

        with pytest.raises(ValueError):
            await interceptor({"message": "hi"})

        assert len(exporter.records) == 1


# ---------------------------------------------------------------------------
# NodeInterceptor — timeout wrapping (C1)
# ---------------------------------------------------------------------------

class TestNodeTimeout:
    @pytest.mark.asyncio
    async def test_node_timeout_is_recorded_and_reraised(self):
        exporter = RecordingExporter()

        async def slow_node(state):
            await asyncio.sleep(10)
            return state

        interceptor = NodeInterceptor(
            slow_node,
            node_id="slow",
            config=ConntrailConfig(exporter=exporter, sample_rate=0.0, timeout_seconds=0.05),
        )

        with pytest.raises(asyncio.TimeoutError):
            await interceptor({"message": "hi"})

        assert len(exporter.records) == 1
        record = exporter.records[0]
        assert record.status == "error"
        assert record.error_type == "timeout"
        assert record.stability == "fragile"
        assert record.entropy_score == 1.0

    @pytest.mark.asyncio
    async def test_node_timeout_skips_contrast_analysis(self, mocker):
        exporter = RecordingExporter()
        analyse_spy = mocker.patch("conntrail.analyser.DivergenceAnalyser.analyse")
        generate_spy = mocker.patch("conntrail.contrast.ContrastGenerator.generate")

        async def slow_node(state):
            await asyncio.sleep(10)
            return state

        interceptor = NodeInterceptor(
            slow_node,
            node_id="slow",
            config=ConntrailConfig(exporter=exporter, sample_rate=1.0, timeout_seconds=0.05),
        )

        with pytest.raises(asyncio.TimeoutError):
            await interceptor({"message": "hi"})

        analyse_spy.assert_not_called()
        generate_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_timeout_configured_preserves_existing_behavior(self):
        """timeout_seconds=None (default) — a slow-ish node still completes normally."""
        exporter = RecordingExporter()

        async def quick_node(state):
            await asyncio.sleep(0.01)
            return {**state, "route": "ok"}

        interceptor = NodeInterceptor(
            quick_node, node_id="quick", config=ConntrailConfig(exporter=exporter, sample_rate=0.0)
        )

        output = await interceptor({"message": "hi"})
        assert output["route"] == "ok"
        assert exporter.records == []

    @pytest.mark.asyncio
    async def test_timeout_not_exceeded_behaves_normally(self):
        exporter = RecordingExporter()

        async def quick_node(state):
            await asyncio.sleep(0.01)
            return {**state, "route": "ok"}

        interceptor = NodeInterceptor(
            quick_node,
            node_id="quick",
            config=ConntrailConfig(exporter=exporter, sample_rate=0.0, timeout_seconds=5.0),
        )

        output = await interceptor({"message": "hi"})
        assert output["route"] == "ok"
        assert exporter.records == []
