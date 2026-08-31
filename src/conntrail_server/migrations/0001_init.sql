CREATE TABLE traces (
    trace_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    original_input TEXT NOT NULL,
    original_route TEXT NOT NULL,
    entropy_score REAL NOT NULL,
    stability TEXT NOT NULL,
    attribution_dimension TEXT NOT NULL,
    plain_language_summary TEXT NOT NULL,
    raw_contrasts TEXT NOT NULL,
    raw_outputs TEXT NOT NULL,
    counterfactual_route TEXT,
    status TEXT NOT NULL DEFAULT 'ok',
    error_type TEXT,
    error_message TEXT
);

CREATE INDEX idx_traces_node_id ON traces(node_id);
CREATE INDEX idx_traces_timestamp ON traces(timestamp);
CREATE INDEX idx_traces_stability ON traces(stability);
CREATE INDEX idx_traces_status ON traces(status);
