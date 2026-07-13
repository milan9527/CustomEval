"""OTEL conformance coverage for `saes doctor` (SPEC §7.1a, T12).

Samples span dicts from a dump and reports which GenAI-convention attributes
and grouping keys are present, so a third party sees exactly what their
instrumentation is missing before running an evaluation (rather than getting
silently empty sessions).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Attribute keys the native mappers rely on, grouped by purpose. Each entry is a
# list of accepted aliases (any present counts as covered). An alias ending in
# `*` is a prefix wildcard, matching indexed conventions like
# `gen_ai.prompt.0.content` or `llm.input_messages.0.message.content`.
# Aliases span the conventions the native mappers actually accept: OTEL GenAI,
# OpenInference (`input.value`, `llm.*_messages.*`), and traceloop / LangChain-
# OTEL (`traceloop.entity.*`, indexed `gen_ai.*`).
_CHECKS: dict[str, list[str]] = {
    "session id": ["session.id", "gen_ai.session.id", "session_id"],
    "scope name (mapper selection)": ["scope.name"],
    "prompt / input": [
        "gen_ai.prompt",
        "gen_ai.prompt.*",
        "gen_ai.input.messages",
        "input.value",
        "llm.input_messages.*",
        "traceloop.entity.input",
    ],
    "completion / output": [
        "gen_ai.completion",
        "gen_ai.completion.*",
        "gen_ai.output.messages",
        "output.value",
        "llm.output_messages.*",
        "traceloop.entity.output",
    ],
    "tool name": [
        "gen_ai.tool.name",
        "tool.name",
        "tool_call.function.name",
    ],
    "trace id": ["traceId", "trace_id"],
    "span id": ["spanId", "span_id"],
}


def _alias_present(flat: dict, alias: str) -> bool:
    """True if `alias` (exact, or `prefix*` wildcard) is present with a value."""
    if alias.endswith("*"):
        prefix = alias[:-1]
        return any(k.startswith(prefix) and v not in (None, "") for k, v in flat.items())
    return alias in flat and flat[alias] not in (None, "")


@dataclass
class FieldCoverage:
    label: str
    aliases: list[str]
    present: int
    total: int

    @property
    def covered(self) -> bool:
        return self.present > 0


@dataclass
class ConformanceReport:
    n_spans: int
    fields: list[FieldCoverage] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        # session id + some content are the minimum to reconstruct anything
        by_label = {f.label: f for f in self.fields}
        return (
            self.n_spans > 0
            and by_label["session id"].covered
            and (
                by_label["prompt / input"].covered
                or by_label["completion / output"].covered
            )
        )


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    scope = raw.get("scope")
    if isinstance(scope, dict) and scope.get("name"):
        flat["scope.name"] = scope["name"]
    for key in ("traceId", "trace_id", "spanId", "span_id"):
        if raw.get(key) is not None:
            flat[key] = raw[key]
    attrs = raw.get("attributes")
    if isinstance(attrs, dict):
        flat.update(attrs)
    elif isinstance(attrs, list):
        for item in attrs:
            if isinstance(item, dict) and "key" in item:
                val = item.get("value")
                if isinstance(val, dict):
                    val = next(iter(val.values()), None)
                flat[item["key"]] = val
    return flat


def _read_spans(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text().strip()
    if not text:
        return []
    spans: list[dict[str, Any]] = []
    if "\n" in text and not text.lstrip().startswith("["):
        for line in text.splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                spans.extend(_unwrap(obj))
    else:
        spans.extend(_unwrap(json.loads(text)))
    return spans


def _unwrap(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [s for s in data if isinstance(s, dict)]
    if isinstance(data, dict):
        if "resourceSpans" in data:
            out: list[dict[str, Any]] = []
            for rs in data.get("resourceSpans", []):
                for ss in rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []:
                    out.extend(ss.get("spans", []))
            return out
        if "spans" in data:
            return list(data["spans"])
        return [data]
    return []


def check_conformance(path: str | Path) -> ConformanceReport:
    spans = _read_spans(path)
    flats = [_flatten(s) for s in spans]
    fields: list[FieldCoverage] = []
    for label, aliases in _CHECKS.items():
        present = sum(
            1 for f in flats if any(_alias_present(f, a) for a in aliases)
        )
        fields.append(
            FieldCoverage(label=label, aliases=aliases, present=present, total=len(flats))
        )
    return ConformanceReport(n_spans=len(spans), fields=fields)


__all__ = ["ConformanceReport", "FieldCoverage", "check_conformance"]
