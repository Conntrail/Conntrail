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
    "token_usage",
    "cost_usd",
    "latency_ms",
    "analysis_overhead",
    "cost_findings",
)

_JSON_COLUMNS = {"raw_contrasts", "raw_outputs", "token_usage", "analysis_overhead", "cost_findings"}

_GEPA_COLUMNS = (
    "attempt_id",
    "run_id",
    "prompt_candidate",
    "scalar_score",
    "traces",
    "token_usage",
    "cost_usd",
    "latency_ms",
)

_GEPA_JSON_COLUMNS = {"traces", "token_usage"}


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
            # Cost columns are nullable (legacy rows store NULL, new rows may
            # store JSON "null") — decode only real JSON strings.
            if data.get(col) is not None:
                data[col] = json.loads(data[col])
        return data

    # -- GEPA attempts (G3) --------------------------------------------------

    def insert_gepa_attempt(self, record: dict[str, Any]) -> str:
        """Upsert a PromptAttemptRecord-shaped payload (+ run_id). Returns attempt_id.

        Upsert, not insert-only: GEPA legitimately re-scores the same attempt
        (e.g. re-evaluating an accepted candidate against the full valset
        right after minibatch acceptance) without a fresh begin/end_attempt
        pair, so the same attempt_id can arrive more than once — a plain
        INSERT would raise a UNIQUE-constraint error on the second call.
        """
        row = {col: record.get(col) for col in _GEPA_COLUMNS}
        for col in _GEPA_JSON_COLUMNS:
            row[col] = json.dumps(row[col])
        cols = ", ".join(_GEPA_COLUMNS)
        placeholders = ", ".join(f":{c}" for c in _GEPA_COLUMNS)
        updates = ", ".join(f"{c} = excluded.{c}" for c in _GEPA_COLUMNS if c != "attempt_id")
        with self._conn:
            self._conn.execute(
                f"INSERT INTO gepa_attempts ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(attempt_id) DO UPDATE SET {updates}",
                row,
            )
        return row["attempt_id"]

    def list_gepa_attempts(self, run_id: str) -> list[dict[str, Any]]:
        """List all attempts for a run, in the order they were inserted."""
        cur = self._conn.execute(
            "SELECT * FROM gepa_attempts WHERE run_id = ? ORDER BY rowid ASC", (run_id,)
        )
        return [self._gepa_row_to_dict(row) for row in cur.fetchall()]

    @staticmethod
    def _gepa_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for col in _GEPA_JSON_COLUMNS:
            if data.get(col) is not None:
                data[col] = json.loads(data[col])
        return data

    # -- cost summary (aggregated cost telemetry per node) -------------------

    def cost_summary(self, *, limit: int = 1000) -> dict[str, Any]:
        """Aggregate cost telemetry per node over the most recent traces.

        Returns {"nodes": [...], "shared_prompt_blocks": [...]}:
          - nodes: per node_id — trace/LLM-call counts, token totals, cache
            hit ratio, total/mean cost, mean latency, cost-warning count.
          - shared_prompt_blocks: instruction-block hashes observed on 2+
            DIFFERENT nodes — cross-node repeated instructions, the shared
            cached-prefix candidates.

        Sampled over the most recent ``limit`` traces (limit-based, no count
        endpoint — same caveat as the dashboard failure view).
        """
        cur = self._conn.execute(
            "SELECT node_id, token_usage, cost_usd, latency_ms, cost_findings "
            "FROM traces ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        nodes: dict[str, dict[str, Any]] = {}
        hash_node_ids: dict[str, set[str]] = {}
        hash_occurrences: dict[str, int] = {}

        for row in cur.fetchall():
            node_id = row["node_id"]
            agg = nodes.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "trace_count": 0,
                    "llm_call_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_input_tokens": 0,
                    "cache_write_tokens": 0,
                    "total_cost_usd": 0.0,
                    "cost_reported_traces": 0,
                    "latency_ms_total": 0.0,
                    "latency_reported_traces": 0,
                    "cost_warning_count": 0,
                },
            )
            agg["trace_count"] += 1

            cost = row["cost_usd"]
            if cost is not None:
                agg["total_cost_usd"] += cost
                agg["cost_reported_traces"] += 1
            latency = row["latency_ms"]
            if latency is not None:
                agg["latency_ms_total"] += latency
                agg["latency_reported_traces"] += 1

            usage = row["token_usage"]
            if usage is not None:
                usage = json.loads(usage)
            if isinstance(usage, dict):
                agg["llm_call_count"] += usage.get("llm_call_count") or 0
                agg["input_tokens"] += usage.get("input_tokens") or 0
                agg["output_tokens"] += usage.get("output_tokens") or 0
                agg["cached_input_tokens"] += usage.get("cached_input_tokens") or 0
                agg["cache_write_tokens"] += usage.get("cache_write_tokens") or 0
                for h in usage.get("prompt_hashes") or []:
                    if isinstance(h, str):
                        hash_node_ids.setdefault(h, set()).add(node_id)
                        hash_occurrences[h] = hash_occurrences.get(h, 0) + 1

            findings = row["cost_findings"]
            if findings is not None:
                findings = json.loads(findings)
            if isinstance(findings, list):
                agg["cost_warning_count"] += sum(
                    1
                    for f in findings
                    if isinstance(f, dict) and f.get("severity") == "warning"
                )

        for agg in nodes.values():
            agg["total_cost_usd"] = round(agg["total_cost_usd"], 6)
            agg["mean_cost_usd"] = (
                round(agg["total_cost_usd"] / agg["cost_reported_traces"], 6)
                if agg["cost_reported_traces"]
                else None
            )
            agg["mean_latency_ms"] = (
                round(agg["latency_ms_total"] / agg["latency_reported_traces"], 1)
                if agg["latency_reported_traces"]
                else None
            )
            agg["cache_hit_ratio"] = (
                round(agg["cached_input_tokens"] / agg["input_tokens"], 3)
                if agg["input_tokens"]
                else None
            )
            del agg["cost_reported_traces"]
            del agg["latency_reported_traces"]
            del agg["latency_ms_total"]

        shared = [
            {"hash": h, "node_ids": sorted(node_ids), "occurrences": hash_occurrences[h]}
            for h, node_ids in hash_node_ids.items()
            if len(node_ids) >= 2
        ]
        shared.sort(key=lambda b: b["occurrences"], reverse=True)

        return {"nodes": list(nodes.values()), "shared_prompt_blocks": shared[:20]}
