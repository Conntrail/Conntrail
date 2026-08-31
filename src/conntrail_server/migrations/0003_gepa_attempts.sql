CREATE TABLE gepa_attempts (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    prompt_candidate TEXT NOT NULL,
    scalar_score REAL,
    traces TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_gepa_attempts_run_id ON gepa_attempts(run_id);
