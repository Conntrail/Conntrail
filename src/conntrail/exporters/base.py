"""
BaseExporter — abstract base class for all Conntrail exporters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from conntrail.record import TraceRecord


class BaseExporter(ABC):
    """
    Abstract base class for TraceRecord exporters.

    All exporters must implement write(). The write() method is called
    once per TraceRecord, after analysis completes.
    """

    @abstractmethod
    async def write(self, record: TraceRecord) -> None:
        """
        Persist or display a TraceRecord.

        Args:
            record: The completed trace analysis for one node call.
        """
        ...

    async def close(self) -> None:
        """Optional teardown. Called when the tracing session ends."""
