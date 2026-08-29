"""
Tests for TraceRecord.
"""
import json
from datetime import UTC, datetime

from conntrail.contrast import ContrastSet
from conntrail.record import TraceRecord


def _make_record(**overrides) -> TraceRecord:
    """Build a minimal but complete TraceRecord for testing."""
    defaults = dict(
        trace_id="abc-123",
        node_id="router",
        timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        original_input="I need a refund",
        original_route="refund",
        entropy_score=0.25,
        stability="boundary",
        attribution_dimension="semantic intensity",
        plain_language_summary="The 'router' node routed to 'refund' with boundary confidence.",
        raw_contrasts=ContrastSet(
            similar="I want my money back",
            neutral="I have a question about my order",
            opposite="Great service, no issues!",
        ),
        raw_outputs={
            "original": "refund",
            "similar": "refund",
            "neutral": "general",
            "opposite": "general",
        },
        counterfactual_route=None,
    )
    defaults.update(overrides)
    return TraceRecord(**defaults)


class TestStabilityLabel:
    """stability_label() is a pure function — test it now."""

    def test_zero_entropy_is_confident(self):
        assert TraceRecord.stability_label(0.0) == "confident"

    def test_low_entropy_is_confident(self):
        assert TraceRecord.stability_label(0.24) == "confident"

    def test_boundary_threshold(self):
        assert TraceRecord.stability_label(0.25) == "boundary"

    def test_mid_entropy_is_boundary(self):
        assert TraceRecord.stability_label(0.45) == "boundary"

    def test_high_threshold_is_boundary(self):
        assert TraceRecord.stability_label(0.60) == "boundary"

    def test_above_threshold_is_fragile(self):
        assert TraceRecord.stability_label(0.61) == "fragile"

    def test_max_entropy_is_fragile(self):
        assert TraceRecord.stability_label(1.0) == "fragile"


class TestBuildSummary:
    """build_summary() is a pure function — test it now."""

    def test_basic_summary(self):
        summary = TraceRecord.build_summary(
            node_id="router",
            route="escalate",
            stability="fragile",
            entropy=0.75,
            attribution="urgency",
            counterfactual="general",
        )
        assert "router" in summary
        assert "escalate" in summary
        assert "fragile" in summary
        assert "0.75" in summary
        assert "urgency" in summary
        assert "general" in summary

    def test_summary_no_counterfactual(self):
        summary = TraceRecord.build_summary(
            node_id="router",
            route="general",
            stability="confident",
            entropy=0.1,
            attribution="sentiment",
            counterfactual=None,
        )
        assert "router" in summary
        assert "confident" in summary
        assert "If that dimension" not in summary


class TestTraceRecordSerialization:
    def test_to_dict_returns_dict(self):
        record = _make_record()
        result = record.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_json_serialisable(self):
        record = _make_record()
        # Must not raise
        json.dumps(record.to_dict())

    def test_to_dict_all_fields_present(self):
        record = _make_record()
        d = record.to_dict()
        for key in (
            "trace_id", "node_id", "timestamp", "original_input",
            "original_route", "entropy_score", "stability",
            "attribution_dimension", "plain_language_summary",
            "raw_contrasts", "raw_outputs", "counterfactual_route",
            "status", "error_type", "error_message",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_timestamp_is_isoformat(self):
        record = _make_record()
        ts = record.to_dict()["timestamp"]
        assert isinstance(ts, str)
        # Must parse back without error
        datetime.fromisoformat(ts)

    def test_to_dict_raw_contrasts_is_dict(self):
        record = _make_record()
        contrasts = record.to_dict()["raw_contrasts"]
        assert isinstance(contrasts, dict)
        assert set(contrasts.keys()) == {"similar", "neutral", "opposite"}

    def test_to_dict_counterfactual_none(self):
        record = _make_record(counterfactual_route=None)
        assert record.to_dict()["counterfactual_route"] is None

    def test_to_dict_counterfactual_present(self):
        record = _make_record(counterfactual_route="general")
        assert record.to_dict()["counterfactual_route"] == "general"

    def test_from_dict_roundtrip(self):
        """Serialise → deserialise preserves all fields."""
        original = _make_record()
        restored = TraceRecord.from_dict(original.to_dict())

        assert restored.trace_id == original.trace_id
        assert restored.node_id == original.node_id
        assert restored.timestamp == original.timestamp
        assert restored.original_input == original.original_input
        assert restored.original_route == original.original_route
        assert restored.entropy_score == original.entropy_score
        assert restored.stability == original.stability
        assert restored.attribution_dimension == original.attribution_dimension
        assert restored.plain_language_summary == original.plain_language_summary
        assert restored.raw_contrasts.similar == original.raw_contrasts.similar
        assert restored.raw_contrasts.neutral == original.raw_contrasts.neutral
        assert restored.raw_contrasts.opposite == original.raw_contrasts.opposite
        assert restored.raw_outputs == original.raw_outputs
        assert restored.counterfactual_route == original.counterfactual_route
        assert restored.status == original.status
        assert restored.error_type == original.error_type
        assert restored.error_message == original.error_message

    def test_from_dict_roundtrip_with_counterfactual(self):
        original = _make_record(counterfactual_route="escalation")
        restored = TraceRecord.from_dict(original.to_dict())
        assert restored.counterfactual_route == "escalation"


class TestTraceRecordFailureFields:
    """New in this repo: status/error_type/error_message (needed by F6/C2)."""

    def test_defaults_on_a_normal_success_record(self):
        record = _make_record()
        assert record.status == "ok"
        assert record.error_type is None
        assert record.error_message is None

    def test_success_record_serializes_with_three_extra_keys(self):
        d = _make_record().to_dict()
        assert d["status"] == "ok"
        assert d["error_type"] is None
        assert d["error_message"] is None

    def test_error_record_roundtrip(self):
        original = _make_record(
            status="error",
            error_type="ValueError",
            error_message="boom",
            entropy_score=1.0,
            stability="fragile",
        )
        restored = TraceRecord.from_dict(original.to_dict())
        assert restored.status == "error"
        assert restored.error_type == "ValueError"
        assert restored.error_message == "boom"

    def test_from_dict_defaults_missing_failure_fields(self):
        """Legacy-shaped payloads without the new keys still round-trip."""
        original = _make_record()
        payload = original.to_dict()
        del payload["status"]
        del payload["error_type"]
        del payload["error_message"]
        restored = TraceRecord.from_dict(payload)
        assert restored.status == "ok"
        assert restored.error_type is None
        assert restored.error_message is None
