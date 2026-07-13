"""JSON results sink (SPEC §10.1, T6)."""

from __future__ import annotations

import json
from pathlib import Path

from .build import ReportDocument


def write_json(doc: ReportDocument, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc.to_dict(), indent=2, ensure_ascii=False))
    return path


__all__ = ["write_json"]
