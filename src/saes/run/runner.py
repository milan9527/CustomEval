"""On-demand evaluation runner (SPEC §8.1) — native Experiment orchestration.

SAES owns only the wiring: load sessions (ingest), resolve evaluators + judge
(evaluators/judge), build native Cases, and drive the native
`Experiment.run_evaluations_async`. The Experiment/Case/Report engine, judge
invocation, and scoring are all native `strands-agents-evals`.

Flow (score existing sessions — mirrors AgentCore EvaluationClient):
  1. ingest.load_sessions(config.data_source)  -> [native Session]
  2. for each Session: build a Case(input=session_id, expected_* from ground truth)
  3. task(case) returns {"output": final_response, "trajectory": Session}
  4. Experiment(cases, evaluators).run_evaluations_async(task) -> EvaluationReport
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from strands_evals import Case, Experiment

from ..config.schema import EvaluationConfig
from ..evaluators.registry import resolve_evaluator
from ..ingest import load_sessions
from ..judge.providers import build_model
from .ground_truth import GroundTruthSet, load_ground_truth


@dataclass
class RunResult:
    """Outcome of an on-demand run (SPEC §10.1)."""

    config_name: str
    judge_model: str
    report: Any  # native EvaluationReport
    evaluator_ids: list[str] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
    aggregates: dict[str, dict[str, float]] = field(default_factory=dict)


def _final_output(session: Any) -> str:
    """Final agent response text from a native Session.

    The native Session model carries the response on the AgentInvocationSpan
    (`agent_response`). We walk traces newest-first and return the last agent
    response found. Verified against strands_evals v1.0.2 span shapes.
    """
    traces = getattr(session, "traces", None) or []
    for trace in reversed(traces):
        for span in reversed(getattr(trace, "spans", None) or []):
            resp = getattr(span, "agent_response", None)
            if isinstance(resp, str) and resp.strip():
                return resp.strip()
    return ""


def _session_id(session: Any, fallback: str) -> str:
    return str(getattr(session, "session_id", None) or fallback)


def _build_cases(
    sessions: list[Any], gt: GroundTruthSet
) -> tuple[list[Case], dict[str, Any]]:
    """Build one native Case per session, attaching ground truth. Returns the
    cases and a session_id -> Session map for the task closure."""
    cases: list[Case] = []
    by_id: dict[str, Any] = {}
    for i, session in enumerate(sessions):
        sid = _session_id(session, f"session-{i}")
        by_id[sid] = session
        cases.append(_case_for(sid, gt))
    return cases, by_id


async def run_on_demand(config: EvaluationConfig) -> RunResult:
    """Execute an on-demand evaluation over sessions from the data source."""
    gt = load_ground_truth(config.ground_truth)

    judge_model = build_model(config.judge)
    evaluators = [resolve_evaluator(ref, judge_model) for ref in config.evaluators]
    evaluator_ids = [ref.id for ref in config.evaluators]

    cases, task, session_ids = _build_task(config, gt)

    experiment = Experiment(cases=cases, evaluators=evaluators)
    report = await experiment.run_evaluations_async(task)

    aggregates = _aggregate(report, evaluator_ids)
    return RunResult(
        config_name=config.name,
        judge_model=config.judge.model,
        report=report,
        evaluator_ids=evaluator_ids,
        session_ids=session_ids,
        aggregates=aggregates,
    )


def _build_task(
    config: EvaluationConfig, gt: GroundTruthSet
) -> tuple[list[Case], Any, list[str]]:
    """Resolve the data source into (cases, task_callable, session_ids).

    - cloudwatch: discover session ids, build Cases with ground truth, and use
      the native provider's `as_task()` closure (SAES owns only discovery).
    - otlp_file / other: load native Sessions locally and serve them via a
      closure over the in-memory session map (M1 path).
    """
    if config.data_source.type == "cloudwatch":
        return _cloudwatch_task(config, gt)
    return _local_task(config, gt)


def _local_task(
    config: EvaluationConfig, gt: GroundTruthSet
) -> tuple[list[Case], Any, list[str]]:
    sessions = load_sessions(config.data_source)
    cases, by_id = _build_cases(sessions, gt)

    def task(case: Case) -> dict[str, Any]:
        session = by_id[case.input]
        return {"output": _final_output(session), "trajectory": session}

    return cases, task, list(by_id)


def _cloudwatch_task(
    config: EvaluationConfig, gt: GroundTruthSet
) -> tuple[list[Case], Any, list[str]]:
    from ..ingest.cloudwatch import build_provider, discover_session_ids
    from ..ingest.cloudwatch_task import build_supplemented_task

    provider = build_provider(config.data_source.cloudwatch)
    cw_cfg = config.data_source.cloudwatch
    session_ids = discover_session_ids(provider, cw_cfg)
    cases = [_case_for(sid, gt) for sid in session_ids]
    # native read+map, wrapped with the non-Strands tool-trajectory supplement
    task = build_supplemented_task(provider, cw_cfg)
    return cases, task, session_ids


def _case_for(sid: str, gt: GroundTruthSet) -> Case:
    ref = gt.for_session(sid)
    return Case(
        name=sid,
        session_id=sid,
        input=sid,
        expected_output=ref.expected_response,
        expected_assertion=ref.assertions,
        expected_trajectory=ref.expected_trajectory,
    )


def _aggregate(report: Any, evaluator_ids: list[str]) -> dict[str, dict[str, float]]:
    """Per-evaluator aggregates (avg, pass_rate, n, errored) from the native
    report.

    The native report flattens results to one row per (evaluator, case) pair:
    `detailed_results[i]` (a one-element list of EvaluationOutput) aligns with
    `report.cases[i]`, whose `evaluator` field names the evaluator (SAES sets
    this to the AgentCore-style id via the registry). We group by that name so
    aggregation is robust to the flattened, evaluator-major ordering.
    """
    detailed = getattr(report, "detailed_results", None) or []
    cases = getattr(report, "cases", None) or []

    acc: dict[str, dict[str, float]] = {
        ev_id: {"sum": 0.0, "passes": 0.0, "n": 0.0, "errored": 0.0}
        for ev_id in evaluator_ids
    }

    for i, outputs in enumerate(detailed):
        ev_id = _evaluator_of(cases, i)
        bucket = acc.get(ev_id)
        if bucket is None:
            continue  # evaluator not in our config list (defensive)
        for out in outputs:
            if getattr(out, "label", None) == "ERROR":
                bucket["errored"] += 1
                continue
            score = getattr(out, "score", None)
            if score is None:
                continue
            bucket["sum"] += float(score)
            bucket["n"] += 1
            if getattr(out, "test_pass", False):
                bucket["passes"] += 1

    result: dict[str, dict[str, float]] = {}
    for ev_id in evaluator_ids:
        b = acc[ev_id]
        n = b["n"]
        result[ev_id] = {
            "avg": (b["sum"] / n) if n else 0.0,
            "pass_rate": (b["passes"] / n) if n else 0.0,
            "n": n,
            "errored": b["errored"],
        }
    return result


def _evaluator_of(cases: list[Any], i: int) -> str | None:
    """Extract the evaluator name/id for result row i from report.cases."""
    if i >= len(cases):
        return None
    case = cases[i]
    if isinstance(case, dict):
        return case.get("evaluator")
    return getattr(case, "evaluator", None)


__all__ = ["RunResult", "run_on_demand"]
