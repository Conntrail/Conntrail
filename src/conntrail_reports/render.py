"""
Render an AuditData view model to a self-contained HTML report.

The template and stylesheet are embedded so a report is a single portable
file. If WeasyPrint is installed, :func:`render_pdf` converts the same HTML to
a PDF; otherwise callers can open the HTML and print to PDF from the browser.
"""
from __future__ import annotations

from pathlib import Path

from conntrail_reports.data import AuditData

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _jinja_env():
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env


def render_html(data: AuditData) -> str:
    """Render the cost-audit report to a self-contained HTML string."""
    env = _jinja_env()
    css = (_TEMPLATE_DIR / "report.css").read_text(encoding="utf-8")
    template = env.get_template("cost_audit.html")
    return template.render(data=data, css=css)


def render_pdf(data: AuditData, out_path: str | Path) -> Path:
    """Render the report to PDF via WeasyPrint. Raises ImportError if absent."""
    from weasyprint import HTML  # type: ignore[import-not-found]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=render_html(data)).write_pdf(str(out))
    return out
