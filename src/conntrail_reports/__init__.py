"""
Conntrail reports — derive audit reports from collector data.

This package is a presentation/export layer only. It consumes the aggregates
the collector already produces (``/v1/cost-summary``) and the per-trace
records, and renders them. It never invents findings and never auto-applies
changes — findings recommend, the human decides.
"""
from conntrail_reports.data import (
    AuditData,
    FindingRollup,
    audit_data_from_bundle,
    build_audit_data,
)
from conntrail_reports.render import render_html, render_pdf

__all__ = [
    "AuditData",
    "FindingRollup",
    "build_audit_data",
    "audit_data_from_bundle",
    "render_html",
    "render_pdf",
]
