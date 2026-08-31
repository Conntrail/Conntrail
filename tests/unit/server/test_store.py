"""
Tests for TraceStore (sqlite schema + repository).
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from conntrail.contrast import ContrastSet
from conntrail.record import TraceRecord
from conntrail_server.migrations import run_migrations
from conntrail_server.store import TraceStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    node_id: str = "router",
    timestamp: datetime | None = None,
    stability: str = "confident",
    status: str = "ok",
    entropy_score: float = 0.1,
) -> dict:
    record = TraceRecord(
        trace_id=TraceRecord.make_id(),
        node_id=node_id,
        timestamp=timestamp or datetime.now(UTC),
        original_input="I need a refund",
        original_route="refund",
        entropy_score=entropy_score,
        stability=stability,
        attribution_dimension="urgency",
        plain_language_summary="The 'router' node routed to 'refund' with confident confidence.",
        raw_contrasts=ContrastSet(similar="a", neutral="b", opposite="c"),
        raw_outputs={"similar": "refund", "neutral": "refund", "opposite": "escalation"},
        counterfactual_route="escalation",
        status=status,
        error_type=None if status == "ok" else "ValueError",
        error_message=None if status == "ok" else "boom",
    )
    return record.to_dict()


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "traces.sqlite3"
    s = TraceStore(db_path)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# insert / get round trip
# ---------------------------------------------------------------------------


def test_insert_returns_trace_id(store):
    payload = _make_record()
    trace_id = store.insert(payload)
    assert trace_id == payload["trace_id"]


def test_insert_then_get_round_trips_exact_payload(store):
    payload = _make_record()
    store.insert(payload)
    fetched = store.get(payload["trace_id"])
    # failure_category is a store-level column populated by the ingest layer
    # (C2), not part of TraceRecord.to_dict()'s wire shape — absent here.
    assert fetched == {**payload, "failure_category": None}


def test_insert_then_get_round_trips_error_record(store):
    payload = _make_record(status="error")
    store.insert(payload)
    fetched = store.get(payload["trace_id"])
    assert fetched == {**payload, "failure_category": None}
    assert fetched["error_type"] == "ValueError"
    assert fetched["error_message"] == "boom"


def test_get_unknown_trace_id_returns_none(store):
    assert store.get("does-not-exist") is None


def test_raw_contrasts_and_raw_outputs_round_trip_as_dicts(store):
    payload = _make_record()
    store.insert(payload)
    fetched = store.get(payload["trace_id"])
    assert fetched["raw_contrasts"] == {"similar": "a", "neutral": "b", "opposite": "c"}
    assert fetched["raw_outputs"] == payload["raw_outputs"]


# ---------------------------------------------------------------------------
# list() filtering
# ---------------------------------------------------------------------------


def test_list_filters_by_node_id(store):
    p1 = _make_record(node_id="router_a")
    p2 = _make_record(node_id="router_b")
    store.insert(p1)
    store.insert(p2)

    results = store.list(node_id="router_a")
    assert [r["trace_id"] for r in results] == [p1["trace_id"]]


def test_list_filters_by_stability(store):
    p1 = _make_record(stability="fragile")
    p2 = _make_record(stability="confident")
    store.insert(p1)
    store.insert(p2)

    results = store.list(stability="fragile")
    assert [r["trace_id"] for r in results] == [p1["trace_id"]]


def test_list_filters_by_status(store):
    p1 = _make_record(status="error")
    p2 = _make_record(status="ok")
    store.insert(p1)
    store.insert(p2)

    results = store.list(status="error")
    assert [r["trace_id"] for r in results] == [p1["trace_id"]]


def test_list_filters_by_since_and_until(store):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    p_old = _make_record(timestamp=base - timedelta(days=2))
    p_mid = _make_record(timestamp=base)
    p_new = _make_record(timestamp=base + timedelta(days=2))
    for p in (p_old, p_mid, p_new):
        store.insert(p)

    since = (base - timedelta(days=1)).isoformat()
    until = (base + timedelta(days=1)).isoformat()
    results = store.list(since=since, until=until)
    assert [r["trace_id"] for r in results] == [p_mid["trace_id"]]


def test_list_combines_multiple_filters(store):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    match = _make_record(node_id="router", stability="fragile", timestamp=base)
    wrong_node = _make_record(node_id="other", stability="fragile", timestamp=base)
    wrong_stability = _make_record(node_id="router", stability="confident", timestamp=base)
    for p in (match, wrong_node, wrong_stability):
        store.insert(p)

    results = store.list(node_id="router", stability="fragile")
    assert [r["trace_id"] for r in results] == [match["trace_id"]]


def test_list_empty_result(store):
    store.insert(_make_record(node_id="router"))
    assert store.list(node_id="does-not-exist") == []


# ---------------------------------------------------------------------------
# list() pagination
# ---------------------------------------------------------------------------


def test_list_pagination_limit_and_offset(store):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    payloads = [
        _make_record(node_id="router", timestamp=base + timedelta(minutes=i)) for i in range(5)
    ]
    for p in payloads:
        store.insert(p)

    # newest first
    expected_order = [p["trace_id"] for p in reversed(payloads)]

    page1 = store.list(node_id="router", limit=2, offset=0)
    page2 = store.list(node_id="router", limit=2, offset=2)
    page3 = store.list(node_id="router", limit=2, offset=4)

    assert [r["trace_id"] for r in page1] == expected_order[0:2]
    assert [r["trace_id"] for r in page2] == expected_order[2:4]
    assert [r["trace_id"] for r in page3] == expected_order[4:5]


# ---------------------------------------------------------------------------
# migration runner
# ---------------------------------------------------------------------------


def test_migration_runner_is_idempotent(tmp_path):
    db_path = tmp_path / "idempotent.sqlite3"
    conn = sqlite3.connect(str(db_path))
    from conntrail_server.store import _MIGRATIONS_DIR

    run_migrations(conn, _MIGRATIONS_DIR)
    run_migrations(conn, _MIGRATIONS_DIR)  # must not raise or duplicate schema

    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "traces" in tables

    applied = conn.execute("SELECT filename FROM schema_migrations").fetchall()
    assert len(applied) == len(list(_MIGRATIONS_DIR.glob("*.sql")))
    conn.close()


def test_store_construction_runs_migrations(tmp_path):
    db_path = tmp_path / "fresh.sqlite3"
    store = TraceStore(db_path)
    try:
        # Should not raise — table exists and is queryable.
        assert store.list() == []
    finally:
        store.close()
