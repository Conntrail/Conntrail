"""
RecordingExporter — an in-memory BaseExporter for tests, CI gates, and fixture
capture.

Use it when you want the TraceRecords but do not want to run a collector:
inject it as ``ConntrailConfig.exporter`` and read ``records`` after the
traced call(s) return.
"""
from __future__ import annotations

from conntrail.exporters.base import BaseExporter
from conntrail.record import TraceRecord


class RecordingExporter(BaseExporter):
    """Collects every written TraceRecord in memory, in write order."""

    def __init__(self) -> None:
        self.records: list[TraceRecord] = []

    async def write(self, record: TraceRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()
