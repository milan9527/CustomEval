"""T6 report tests — build, JSON, HTML."""

import json

from saes.report import build_report, render_html, write_html, write_json
from saes.run.gate import evaluate_gate
from saes.run.runner import RunResult


class _Out:
    def __init__(self, score, reason, label=None, test_pass=True):
        self.score = score
        self.reason = reason
        self.label = label
        self.test_pass = test_pass


class _Report:
    def __init__(self, rows, overall=0.0):
        self.detailed_results = [[out] for _, _, out in rows]
        self.cases = [{"evaluator": ev, "name": sid} for ev, sid, _ in rows]
        self.overall_score = overall


def _run_result():
    report = _Report(
        [
            ("Builtin.Helpfulness", "s1", _Out(0.9, "moved user forward", "Very Helpful")),
            ("Builtin.Helpfulness", "s2", _Out(0.4, "some confusion", "Somewhat Unhelpful")),
        ],
        overall=0.65,
    )
    return RunResult(
        config_name="demo",
        judge_model="gpt-4.1",
        report=report,
        evaluator_ids=["Builtin.Helpfulness"],
        session_ids=["s1", "s2"],
        aggregates={
            "Builtin.Helpfulness": {"avg": 0.65, "pass_rate": 0.5, "n": 2.0, "errored": 0.0}
        },
    )


def test_build_report_flattens_rows():
    doc = build_report(_run_result())
    assert doc.config_name == "demo"
    assert doc.judge_model == "gpt-4.1"
    assert len(doc.rows) == 2
    r0 = doc.rows[0]
    assert r0.evaluator_id == "Builtin.Helpfulness"
    assert r0.session_id == "s1"
    assert r0.reason == "moved user forward"
    assert r0.label == "Very Helpful"


def test_build_report_with_gate():
    gate = evaluate_gate(
        ["Builtin.Helpfulness.avg >= 0.8"],
        {"Builtin.Helpfulness": {"avg": 0.65, "pass_rate": 0.5, "n": 2.0, "errored": 0.0}},
    )
    doc = build_report(_run_result(), gate=gate)
    assert doc.gate is not None
    assert doc.gate["passed"] is False
    assert doc.gate["checks"][0]["actual"] == 0.65


def test_write_json_roundtrip(tmp_path):
    doc = build_report(_run_result())
    p = write_json(doc, tmp_path / "out" / "results.json")
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["config_name"] == "demo"
    assert len(data["rows"]) == 2
    assert data["aggregates"]["Builtin.Helpfulness"]["avg"] == 0.65


def test_render_html_contains_reasoning_and_scores():
    doc = build_report(_run_result())
    html = render_html(doc)
    assert "SAES Evaluation Report" in html
    assert "Builtin.Helpfulness" in html
    assert "moved user forward" in html  # judge reasoning drill-down
    assert "gpt-4.1" in html


def test_html_escapes_reasoning(tmp_path):
    # a reason containing HTML must be escaped, not injected
    rr = _run_result()
    rr.report.detailed_results[0][0].reason = "<script>alert(1)</script>"
    doc = build_report(rr)
    html = render_html(doc)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_write_html_file(tmp_path):
    doc = build_report(_run_result())
    p = write_html(doc, tmp_path / "report.html")
    assert p.exists()
    assert "<!DOCTYPE html>" in p.read_text()


def test_html_shows_gate_status():
    gate = evaluate_gate(
        ["Builtin.Helpfulness.avg >= 0.8"],
        {"Builtin.Helpfulness": {"avg": 0.65, "pass_rate": 0.5, "n": 2.0, "errored": 0.0}},
    )
    html = render_html(build_report(_run_result(), gate=gate))
    assert "GATE FAILED" in html
