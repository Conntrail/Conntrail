"""
Simple sequential-file migration runner for the sqlite trace store.

No ORM, no Alembic — migrations are plain .sql files applied in filename
order and tracked in a schema_migrations table. Safe to call repeatedly;
already-applied migrations are skipped.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

_TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def run_migrations(conn: sqlite3.Connection, migrations_dir: Path) -> None:
    """Apply any .sql files in migrations_dir not yet recorded as applied."""
    conn.execute(_TRACKING_TABLE)
    applied = {row[0] for row in conn.execute("SELECT filename FROM schema_migrations")}
    for path in sorted(migrations_dir.glob("*.sql")):
        if path.name in applied:
            continue
        conn.executescript(path.read_text())
        conn.execute("INSERT INTO schema_migrations (filename) VALUES (?)", (path.name,))
    conn.commit()
