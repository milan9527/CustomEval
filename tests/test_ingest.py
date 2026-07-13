"""Ingestion tests — SAES's own seam over native strands_evals mappers.

SAES owns: reading a local dump, grouping spans by session id, selecting the
native mapper (auto or explicit), and delegating. It does NOT own the mapping
logic itself. So we test the seam with a stub mapper (deterministic, no
dependence on native span-shape internals), plus one real smoke test that
delegation returns a native `Session`.
"""

from pathlib import Path

import pytest

from saes.config.schema import DataSourceConfig
from saes.ingest import load_sessions, load_sessions_from_file

FIXTURES = Path(__file__).parent / "fixtures"


class _StubMapper:
    """Records the (spans, session_id) it was asked to map."""

    def __init__(self):
        self.calls = []

    def map_to_session(self, data, session_id):
        self.calls.append((session_id, list(data)))
        return {"session_id": session_id, "n_spans": len(data)}


# ---- SAES seam: grouping + delegation (stub mapper) -------------------------

def test_groups_spans_by_session_id():
    stub = _StubMapper()
    sessions = load_sessions_from_file(FIXTURES / "otel_session.jsonl", mapper=stub)
    # otel fixture has one session id "sess-abc" across 3 spans
    assert len(sessions) == 1
    session_id, spans = stub.calls[0]
    assert session_id == "sess-abc"
    assert len(spans) == 3


def test_multiple_sessions_split(tmp_path):
    dump = tmp_path / "multi.jsonl"
    dump.write_text(
        '{"spanId":"a","attributes":{"session.id":"s1"}}\n'
        '{"spanId":"b","attributes":{"session.id":"s2"}}\n'
        '{"spanId":"c","attributes":{"session.id":"s1"}}\n'
    )
    stub = _StubMapper()
    sessions = load_sessions_from_file(dump, mapper=stub)
    assert len(sessions) == 2
    ids = [c[0] for c in stub.calls]
    assert set(ids) == {"s1", "s2"}
    # s1 got 2 spans, s2 got 1
    by_id = {sid: len(spans) for sid, spans in stub.calls}
    assert by_id == {"s1": 2, "s2": 1}


def test_reads_otlp_envelope_shape(tmp_path):
    dump = tmp_path / "otlp.json"
    dump.write_text(
        '{"resourceSpans":[{"scopeSpans":[{"spans":['
        '{"spanId":"x","attributes":{"session.id":"env-s"}}]}]}]}'
    )
    stub = _StubMapper()
    sessions = load_sessions_from_file(dump, mapper=stub)
    assert len(sessions) == 1
    assert stub.calls[0][0] == "env-s"


def test_empty_file_yields_no_sessions(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert load_sessions_from_file(p, mapper=_StubMapper()) == []


def test_missing_session_id_defaults(tmp_path):
    dump = tmp_path / "nosession.jsonl"
    dump.write_text('{"spanId":"a","attributes":{}}\n')
    stub = _StubMapper()
    load_sessions_from_file(dump, mapper=stub)
    assert stub.calls[0][0] == "default-session"


# ---- real delegation smoke test (native mapper) -----------------------------

def test_openinference_dump_maps_to_native_session():
    """A non-Strands, OpenInference-instrumented dump (correct scope) maps to a
    native Session via the auto-detected native mapper — no AWS, no SAES mapping
    code. This is the framework-agnostic guarantee (SPEC §7, D2)."""
    from strands_evals.types.trace import Session

    sessions = load_sessions_from_file(FIXTURES / "openinference_session.jsonl")
    assert len(sessions) >= 1
    assert isinstance(sessions[0], Session)
    assert sessions[0].session_id == "oi-sess"


def test_realistic_openinference_reconstructs_nonempty_trace():
    """T11 — a realistic OpenInference dump (live-instrumentation attribute
    format the native mapper actually requires) reconstructs a NON-EMPTY trace,
    not just a Session object. Upgrades the D2 guarantee from 'returns a
    Session' to 'returns a faithful Session'."""
    from strands_evals.types.trace import Session

    sessions = load_sessions_from_file(FIXTURES / "openinference_real.jsonl")
    assert len(sessions) == 1
    s = sessions[0]
    assert isinstance(s, Session)
    assert s.session_id == "oi-real"
    # the native mapper actually built a trace with an inference span
    assert len(s.traces) == 1
    assert len(s.traces[0].spans) >= 1


def test_langgraph_dump_reconstructs_agent_invocation_span():
    """D2 — a non-Strands LangGraph/LangChain-OTEL (traceloop) dump maps via the
    native LangChainOtelSessionMapper into a Session with an AgentInvocationSpan,
    which is what the TRACE_LEVEL evaluators require. No Strands, no SAES mapping
    code."""
    sessions = load_sessions_from_file(FIXTURES / "langgraph_session.jsonl")
    assert len(sessions) == 1
    s = sessions[0]
    assert s.session_id == "langgraph-sess"
    span_types = {type(sp).__name__ for tr in s.traces for sp in tr.spans}
    assert "AgentInvocationSpan" in span_types


# ---- config-driven source resolution ----------------------------------------

def test_load_sessions_via_config():
    cfg = DataSourceConfig(
        type="otlp_file", path=str(FIXTURES / "openinference_session.jsonl")
    )
    sessions = load_sessions(cfg)
    assert len(sessions) >= 1


def test_explicit_mapper_name_in_config():
    cfg = DataSourceConfig(
        type="otlp_file",
        path=str(FIXTURES / "openinference_session.jsonl"),
        mapper="openinference",
    )
    sessions = load_sessions(cfg)
    assert len(sessions) >= 1


def test_unknown_mapper_name_raises():
    cfg = DataSourceConfig(
        type="otlp_file", path=str(FIXTURES / "otel_session.jsonl"), mapper="bogus"
    )
    with pytest.raises(ValueError, match="unknown mapper"):
        load_sessions(cfg)


def test_cloudwatch_source_deferred_to_m2():
    cfg = DataSourceConfig(
        type="cloudwatch",
        cloudwatch={"log_group_names": ["/aws/x"], "region": "us-east-1"},
    )
    with pytest.raises(NotImplementedError, match="CloudWatchProvider"):
        load_sessions(cfg)
