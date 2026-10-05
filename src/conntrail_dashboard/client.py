"""
CollectorClient — re-exported from the shared :mod:`conntrail_client` package.

Kept as a thin shim so existing dashboard imports
(``from conntrail_dashboard.client import CollectorClient``) keep working while
the implementation lives in one place shared with the report generator.
"""
from __future__ import annotations

from conntrail_client import CollectorClient

__all__ = ["CollectorClient"]
