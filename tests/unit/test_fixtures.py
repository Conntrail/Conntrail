"""
Tests for fixture bundle read/write and harness resolution.
"""
from __future__ import annotations

import pytest

from conntrail.fixtures import load_fixture_bundle, load_harness, write_fixture_bundle
from conntrail.record import TraceRecord
from tests.fixtures.trace_records import make_trace_payload


def _record(node_id: str = "router", entropy: float = 0.1) -> TraceRecord:
    return TraceRecord.from_dict(
        make_trace_payload(node_id=node_id, entropy_score=entropy)
    )


def test_write_then_load_round_trip(tmp_path):
    records = [_record("router", 0.1), _record("classifier", 0.7)]
    out = write_fixture_bundle(records, tmp_path / "nested" / "fixtures.json")

    assert out.exists()
    loaded = load_fixture_bundle(out)
    assert [r["node_id"] for r in loaded] == ["router", "classifier"]
    assert loaded[1]["entropy_score"] == 0.7


def test_load_rejects_non_list_records(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"records": "nope"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_fixture_bundle(path)


def test_load_harness_requires_colon():
    with pytest.raises(ValueError):
        load_harness("conntrail.record")


def test_load_harness_missing_attribute():
    with pytest.raises(ValueError):
        load_harness("conntrail.record:does_not_exist")


def test_load_harness_non_callable():
    with pytest.raises(ValueError):
        load_harness("conntrail:__all__")


def test_load_harness_resolves_callable():
    harness = load_harness("tests.unit.test_fixtures:_sample_harness")
    assert harness() == []


def _sample_harness():
    return []
