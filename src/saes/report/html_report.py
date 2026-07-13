"""Self-contained HTML report (SPEC §10.1, T6)."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .build import ReportDocument

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_html(doc: ReportDocument) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("report.html.j2")
    return template.render(doc=doc)


def write_html(doc: ReportDocument, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(doc))
    return path


__all__ = ["render_html", "write_html"]
