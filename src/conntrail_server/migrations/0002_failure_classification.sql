ALTER TABLE traces ADD COLUMN failure_category TEXT;

CREATE INDEX idx_traces_failure_category ON traces(failure_category);
