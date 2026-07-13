"""Build a serializable report from a RunResult (SPEC §10.1, T6).

Flattens the native EvaluationReport into SAES per-row records (one per
evaluator × session), pairing each EvaluationOutput with its case metadata
(session id, evaluator id) so JSON/HTML can drill into judge reasoning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..run.runner import RunResult


@dataclass
class ResultRow:
    """One evaluator's verdict on one session (SPEC §2.1, trimmed to M1)."""

    evaluator_id: str
    session_id: str
    score: float
    reason: str
    label: str | None = None
    test_pass: bool | None = None
    errored: bool = False


@dataclass
class ReportDocument:
    config_name: str
    judge_model: str
    evaluator_ids: list[str]
    session_ids: list[str]
    overall_score: float
    aggregates: dict[str, dict[str, float]]
    rows: list[ResultRow] = field(default_factory=list)
    gate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _row_from(case: Any, output: Any) -> ResultRow:
    ev_id = case.get("evaluator") if isinstance(case, dict) else getattr(case, "evaluator", "")
    sid = case.get("name") if isinstance(case, dict) else getattr(case, "name", "")
    label = getattr(output, "label", None)
    return ResultRow(
        evaluator_id=str(ev_id or ""),
        session_id=str(sid or ""),
        score=float(getattr(output, "score", 0.0) or 0.0),
        reason=str(getattr(output, "reason", "") or ""),
        label=label,
        test_pass=getattr(output, "test_pass", None),
        errored=label == "ERROR",
    )


def build_report(result: RunResult, gate: Any | None = None) -> ReportDocument:
    report = result.report
    detailed = getattr(report, "detailed_results", None) or []
    cases = getattr(report, "cases", None) or []

    rows: list[ResultRow] = []
    for i, outputs in enumerate(detailed):
        case = cases[i] if i < len(cases) else {}
        for out in outputs:
            rows.append(_row_from(case, out))

    gate_dict = None
    if gate is not None:
        gate_dict = {
            "passed": gate.passed,
            "checks": [
                {
                    "rule": c.rule,
                    "evaluator_id": c.evaluator_id,
                    "metric": c.metric,
                    "op": c.op,
                    "threshold": c.threshold,
                    "actual": c.actual,
                    "passed": c.passed,
                }
                for c in gate.checks
            ],
        }

    return ReportDocument(
        config_name=result.config_name,
        judge_model=result.judge_model,
        evaluator_ids=result.evaluator_ids,
        session_ids=result.session_ids,
        overall_score=float(getattr(report, "overall_score", 0.0) or 0.0),
        aggregates=result.aggregates,
        rows=rows,
        gate=gate_dict,
    )


__all__ = ["ReportDocument", "ResultRow", "build_report"]
