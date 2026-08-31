"""
TraceStore — sqlite-backed repository for TraceRecord payloads.

Records are stored and returned as plain dicts in the same shape as
TraceRecord.to_dict()/from_dict() — the store has no dependency on the
dataclass itself, just its wire shape.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from conntrail_server.migrations import run_migrations

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_COLUMNS = (
    "trace_id",
    "node_id",
    "timestamp",
    "original_input",
    "original_route",
    "entropy_score",
    "stability",
    "attribution_dimension",
    "plain_language_summary",
    "raw_contrasts",
    "raw_outputs",
    "counterfactual_route",
    "status",
    "error_type",
    "error_message",
    "failure_category",
)

_JSON_COLUMNS = {"raw_contrasts", "raw_outputs"}


class TraceStore:
    """Thin repository over sqlite for TraceRecord.to_dict()-shaped payloads."""

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        run_migrations(self._conn, _MIGRATIONS_DIR)

    def close(self) -> None:
        self._conn.close()

    def insert(self, record: dict[str, Any]) -> str:
        """Insert a TraceRecord.to_dict()-shaped payload. Returns the trace_id."""
        row = {col: record.get(col) for col in _COLUMNS}
        for col in _JSON_COLUMNS:
            row[col] = json.dumps(row[col])
        cols = ", ".join(_COLUMNS)
        placeholders = ", ".join(f":{c}" for c in _COLUMNS)
        with self._conn:
            self._conn.execute(f"INSERT INTO traces ({cols}) VALUES ({placeholders})", row)
        return row["trace_id"]

    def get(self, trace_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
        row = cur.fetchone()
        return None if row is None else self._row_to_dict(row)

    def list(
        self,
        *,
        node_id: str | None = None,
        stability: str | None = None,
        status: str | None = None,
        failure_category: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List traces matching all given filters, newest first."""
        clauses = []
        params: list[Any] = []
        if node_id is not None:
            clauses.append("node_id = ?")
            params.append(node_id)
        if stability is not None:
            clauses.append("stability = ?")
            params.append(stability)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if failure_category is not None:
            clauses.append("failure_category = ?")
            params.append(failure_category)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        cur = self._conn.execute(
            f"SELECT * FROM traces {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params,
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for col in _JSON_COLUMNS:
            data[col] = json.loads(data[col])
        return data
