"""Data source factory (SPEC §7) — framework-agnostic, native-mapper based.

SAES does not reimplement trace ingestion. It delegates to native
`strands_evals` providers (CloudWatch/Langfuse/OpenSearch) and session mappers
(OpenInference/LangChainOtel/CloudWatch/Strands, auto-detected). The only
SAES-owned piece is a thin local-file reader so CI/offline runs need no AWS.

Every source yields native `Session` objects grouped by session id, which feed
straight into the native `Experiment` pipeline (SPEC §8).

The integration contract for ANY agent is unchanged: emit OTEL GenAI-convention
spans. Strands, LangGraph, OpenInference-instrumented, or custom — all map here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from strands_evals.mappers import detect_otel_mapper

from ..config.schema import DataSourceConfig


def _iter_raw_spans(data: Any) -> list[dict[str, Any]]:
    """Flatten supported container shapes into a list of span dicts."""
    if isinstance(data, list):
        return [s for s in data if isinstance(s, dict)]
    if isinstance(data, dict):
        if "resourceSpans" in data:
            spans: list[dict[str, Any]] = []
            for rs in data.get("resourceSpans", []):
                scopes = rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []
                for ss in scopes:
                    spans.extend(ss.get("spans", []))
            return spans
        if "spans" in data:
            return list(data["spans"])
        return [data]
    return []


def _read_span_dicts(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text().strip()
    if not text:
        return []
    if "\n" in text and not text.lstrip().startswith("["):
        spans: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                spans.extend(_iter_raw_spans(json.loads(line)))
        if spans:
            return spans
    return _iter_raw_spans(json.loads(text))


def _session_id_of(span: dict[str, Any]) -> str:
    """Best-effort session id from common attribute locations."""
    attrs = span.get("attributes")
    flat: dict[str, Any] = {}
    if isinstance(attrs, dict):
        flat = attrs
    elif isinstance(attrs, list):
        for item in attrs:
            if isinstance(item, dict) and "key" in item:
                val = item.get("value")
                if isinstance(val, dict):
                    val = next(iter(val.values()), None)
                flat[item["key"]] = val
    for key in ("session.id", "gen_ai.session.id", "session_id"):
        if flat.get(key):
            return str(flat[key])
    return "default-session"


def load_sessions_from_file(
    path: str | Path, mapper: Any | None = None
) -> list[Any]:
    """Read a local OTLP/JSONL span dump into native `Session` objects.

    Groups spans by session id, then uses the auto-detected native mapper
    (or an explicitly provided one) to build each `Session`. No AWS required —
    this is the offline/CI path and the framework-agnostic proof: a non-Strands
    dump maps via the native OpenInference/LangChain mappers just the same.
    """
    span_dicts = _read_span_dicts(path)
    if not span_dicts:
        return []

    by_session: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for span in span_dicts:
        sid = _session_id_of(span)
        if sid not in by_session:
            by_session[sid] = []
            order.append(sid)
        by_session[sid].append(span)

    sessions: list[Any] = []
    for sid in order:
        spans = by_session[sid]
        active_mapper = mapper or detect_otel_mapper(spans)
        sessions.append(active_mapper.map_to_session(spans, session_id=sid))
    return sessions


def load_sessions(config: DataSourceConfig) -> list[Any]:
    """Resolve a DataSourceConfig to native `Session` objects.

    - otlp_file : local reader (no AWS)
    - cloudwatch: native CloudWatchProvider (M2)
    - langfuse  : native LangfuseProvider (M2/M4)
    """
    if config.type == "otlp_file":
        mapper = _explicit_mapper(config.mapper)
        return load_sessions_from_file(config.path, mapper=mapper)

    if config.type == "cloudwatch":
        raise NotImplementedError(
            "cloudwatch data source is delegated to the native CloudWatchProvider "
            "in M2; use type 'otlp_file' for offline runs in M1."
        )

    raise NotImplementedError(f"data source type '{config.type}' is not wired yet")


def _explicit_mapper(name: str | None) -> Any | None:
    if not name:
        return None
    from strands_evals import mappers as m

    mapping = {
        "openinference": m.OpenInferenceSessionMapper,
        "langchain_otel": m.LangChainOtelSessionMapper,
        "cloudwatch": m.CloudWatchSessionMapper,
        "strands": m.StrandsInMemorySessionMapper,
    }
    cls = mapping.get(name)
    if cls is None:
        raise ValueError(f"unknown mapper '{name}'; known: {sorted(mapping)}")
    return cls()


__all__ = ["load_sessions", "load_sessions_from_file"]
