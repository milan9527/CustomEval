"""SAES reporting (T6) — JSON + self-contained HTML."""

from .build import ReportDocument, ResultRow, build_report
from .cloudwatch_sink import emit_to_cloudwatch
from .html_report import render_html, write_html
from .json_sink import write_json

__all__ = [
    "ReportDocument",
    "ResultRow",
    "build_report",
    "emit_to_cloudwatch",
    "render_html",
    "write_html",
    "write_json",
]
