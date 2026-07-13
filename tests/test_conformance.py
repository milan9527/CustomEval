"""T12 — OTEL conformance coverage for `saes doctor` (SPEC §7.1a)."""

from pathlib import Path

from saes.ingest.conformance import check_conformance

FIXTURES = Path(__file__).parent / "fixtures"


def test_conformance_on_otel_fixture():
    rep = check_conformance(FIXTURES / "otel_session.jsonl")
    assert rep.n_spans == 3
    by_label = {f.label: f for f in rep.fields}
    assert by_label["session id"].covered  # all spans carry session.id
    assert by_label["prompt / input"].covered
    assert by_label["tool name"].covered
    assert rep.ok


def test_conformance_flags_missing_fields(tmp_path):
    dump = tmp_path / "bare.jsonl"
    # spans with no session id and no gen_ai content
    dump.write_text('{"spanId": "a"}\n{"spanId": "b"}\n')
    rep = check_conformance(dump)
    by_label = {f.label: f for f in rep.fields}
    assert not by_label["session id"].covered
    assert not by_label["prompt / input"].covered
    assert by_label["span id"].covered
    assert not rep.ok  # missing session id -> not conformant


def test_conformance_partial_coverage_counts(tmp_path):
    dump = tmp_path / "partial.jsonl"
    dump.write_text(
        '{"spanId":"a","attributes":{"session.id":"s","gen_ai.prompt":"hi"}}\n'
        '{"spanId":"b","attributes":{}}\n'  # this one missing session id + prompt
    )
    rep = check_conformance(dump)
    by_label = {f.label: f for f in rep.fields}
    assert by_label["session id"].present == 1
    assert by_label["session id"].total == 2
    # at least one span has session id + content -> ok
    assert rep.ok


def test_conformance_empty(tmp_path):
    dump = tmp_path / "empty.jsonl"
    dump.write_text("")
    rep = check_conformance(dump)
    assert rep.n_spans == 0
    assert not rep.ok


def test_conformance_recognizes_traceloop_format(tmp_path):
    """F7 fix — traceloop/LangChain-OTEL attrs (traceloop.entity.*) count as
    prompt/completion coverage, not a false ✗."""
    dump = tmp_path / "tl.jsonl"
    dump.write_text(
        '{"span_id":"wf","attributes":{"session.id":"s","traceloop.span.kind":"workflow",'
        '"traceloop.entity.input":"{\\"x\\":1}","traceloop.entity.output":"{\\"y\\":2}"}}\n'
    )
    rep = check_conformance(dump)
    by = {f.label: f for f in rep.fields}
    assert by["prompt / input"].covered
    assert by["completion / output"].covered


def test_conformance_recognizes_indexed_genai_keys(tmp_path):
    """F7 fix — indexed `gen_ai.prompt.0.content` / OpenInference
    `llm.input_messages.0.message.content` match via prefix wildcard."""
    dump = tmp_path / "idx.jsonl"
    dump.write_text(
        '{"span_id":"a","attributes":{"session.id":"s",'
        '"gen_ai.prompt.0.content":"hi","gen_ai.completion.0.content":"ok"}}\n'
        '{"span_id":"b","attributes":{"session.id":"s2",'
        '"llm.input_messages.0.message.content":"hi","llm.output_messages.0.message.content":"ok"}}\n'
    )
    rep = check_conformance(dump)
    by = {f.label: f for f in rep.fields}
    assert by["prompt / input"].present == 2
    assert by["completion / output"].present == 2


def test_conformance_wildcard_does_not_overmatch(tmp_path):
    """A prefix wildcard must not match unrelated keys or empty values."""
    dump = tmp_path / "neg.jsonl"
    dump.write_text('{"span_id":"a","attributes":{"session.id":"s","gen_ai.prompt.0.content":""}}\n')
    rep = check_conformance(dump)
    by = {f.label: f for f in rep.fields}
    # empty value must not count as covered
    assert by["prompt / input"].present == 0
