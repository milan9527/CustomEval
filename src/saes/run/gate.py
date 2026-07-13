"""CI gate evaluation (SPEC §8.1).

Parses threshold rules like:
    "Builtin.Helpfulness.avg >= 0.8"
    "Builtin.Correctness.pass_rate > 0.9"
against the per-evaluator aggregates produced by the runner, and reports
pass/fail. A failing gate yields a non-zero exit code at the CLI layer.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass

_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
}

# e.g. "Builtin.Helpfulness.avg >= 0.8"
_RULE_RE = re.compile(
    r"^\s*(?P<metric>[A-Za-z0-9_.]+)\s*"
    r"(?P<op>>=|<=|==|!=|>|<)\s*"
    r"(?P<threshold>-?\d+(?:\.\d+)?)\s*$"
)


class GateError(ValueError):
    """A gate rule could not be parsed or references an unknown metric."""


@dataclass
class GateCheck:
    rule: str
    evaluator_id: str
    metric: str
    op: str
    threshold: float
    actual: float
    passed: bool


@dataclass
class GateReport:
    checks: list[GateCheck]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _split_metric(metric: str) -> tuple[str, str]:
    """Split 'Builtin.Helpfulness.avg' into ('Builtin.Helpfulness', 'avg').

    The metric name is the last dotted segment; everything before it is the
    evaluator id (which itself contains a dot, e.g. 'Builtin.Helpfulness')."""
    idx = metric.rfind(".")
    if idx == -1:
        raise GateError(f"gate metric '{metric}' must be '<evaluatorId>.<metric>'")
    return metric[:idx], metric[idx + 1 :]


def evaluate_gate(
    rules: list[str], aggregates: dict[str, dict[str, float]]
) -> GateReport:
    checks: list[GateCheck] = []
    for rule in rules:
        m = _RULE_RE.match(rule)
        if not m:
            raise GateError(f"cannot parse gate rule: {rule!r}")
        evaluator_id, metric = _split_metric(m.group("metric"))
        op = m.group("op")
        threshold = float(m.group("threshold"))

        if evaluator_id not in aggregates:
            raise GateError(
                f"gate references unknown evaluator '{evaluator_id}'; "
                f"available: {sorted(aggregates)}"
            )
        stats = aggregates[evaluator_id]
        if metric not in stats:
            raise GateError(
                f"gate references unknown metric '{metric}' for '{evaluator_id}'; "
                f"available: {sorted(stats)}"
            )
        actual = stats[metric]
        passed = _OPS[op](actual, threshold)
        checks.append(
            GateCheck(
                rule=rule,
                evaluator_id=evaluator_id,
                metric=metric,
                op=op,
                threshold=threshold,
                actual=actual,
                passed=passed,
            )
        )
    return GateReport(checks=checks)


__all__ = ["GateCheck", "GateError", "GateReport", "evaluate_gate"]
