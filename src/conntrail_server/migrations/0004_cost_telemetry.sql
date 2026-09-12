-- 0004: cost telemetry columns for traces and gepa_attempts.
-- All nullable — legacy rows and capture_cost=False traces store NULL.
ALTER TABLE traces ADD COLUMN token_usage TEXT;
ALTER TABLE traces ADD COLUMN cost_usd REAL;
ALTER TABLE traces ADD COLUMN latency_ms REAL;
ALTER TABLE traces ADD COLUMN analysis_overhead TEXT;
ALTER TABLE traces ADD COLUMN cost_findings TEXT;

ALTER TABLE gepa_attempts ADD COLUMN token_usage TEXT;
ALTER TABLE gepa_attempts ADD COLUMN cost_usd REAL;
ALTER TABLE gepa_attempts ADD COLUMN latency_ms REAL;
