"""T5 gate tests (SPEC §8.1)."""

import pytest

from saes.run import GateError, evaluate_gate

AGG = {
    "Builtin.Helpfulness": {"avg": 0.85, "pass_rate": 0.9, "n": 10.0, "errored": 0.0},
    "Builtin.Correctness": {"avg": 0.72, "pass_rate": 0.6, "n": 10.0, "errored": 1.0},
}


def test_gate_passes_when_all_thresholds_met():
    report = evaluate_gate(
        ["Builtin.Helpfulness.avg >= 0.8", "Builtin.Correctness.avg >= 0.7"], AGG
    )
    assert report.passed is True
    assert len(report.checks) == 2


def test_gate_fails_when_threshold_missed():
    report = evaluate_gate(["Builtin.Correctness.avg >= 0.9"], AGG)
    assert report.passed is False
    assert report.checks[0].actual == 0.72
    assert report.checks[0].threshold == 0.9


def test_gate_all_operators():
    rules = [
        "Builtin.Helpfulness.avg > 0.8",
        "Builtin.Correctness.errored <= 1",
        "Builtin.Helpfulness.pass_rate == 0.9",
        "Builtin.Correctness.avg != 0.9",
        "Builtin.Correctness.avg < 0.8",
    ]
    report = evaluate_gate(rules, AGG)
    assert report.passed is True


def test_gate_unparseable_rule():
    with pytest.raises(GateError, match="cannot parse"):
        evaluate_gate(["helpfulness is good"], AGG)


def test_gate_unknown_evaluator():
    with pytest.raises(GateError, match="unknown evaluator"):
        evaluate_gate(["Builtin.Nope.avg >= 0.5"], AGG)


def test_gate_unknown_metric():
    with pytest.raises(GateError, match="unknown metric"):
        evaluate_gate(["Builtin.Helpfulness.median >= 0.5"], AGG)


def test_gate_metric_without_evaluator_dot():
    with pytest.raises(GateError, match="<evaluatorId>.<metric>"):
        evaluate_gate(["avg >= 0.5"], AGG)
