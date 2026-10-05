"""
Fixture bundles — freeze TraceRecords to JSON for the static website demo and
for offline CI gates.

A bundle is a JSON object::

    {"generated_at": "<iso>", "records": [ <TraceRecord.to_dict()>, ... ]}

The static demo reads these precomputed records so it makes no live LLM calls.
"""
from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from conntrail.record import TraceRecord


def write_fixture_bundle(records: list[TraceRecord], path: str | Path) -> Path:
    """Write records as a JSON fixture bundle, creating parent dirs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "records": [record.to_dict() for record in records],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def load_fixture_bundle(path: str | Path) -> list[dict]:
    """Load a fixture bundle and return its raw record dicts."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"{path}: 'records' must be a list")
    return records


def load_harness(target: str) -> Callable[[], list[TraceRecord]]:
    """Resolve a ``module:callable`` harness that returns a list of TraceRecords."""
    if ":" not in target:
        raise ValueError(f"harness must be 'module:callable', got {target!r}")
    module_name, attr = target.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        harness = getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"{module_name!r} has no attribute {attr!r}") from exc
    if not callable(harness):
        raise ValueError(f"{target!r} is not callable")
    return harness
