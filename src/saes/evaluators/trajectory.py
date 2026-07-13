"""Trajectory ground-truth scorers (SPEC §4.2, §5, T13).

AgentCore exposes three deterministic trajectory scorers that compare the
agent's actual tool sequence against an expected one:
  Builtin.TrajectoryExactOrderMatch  — same tools, same order, no extras
  Builtin.TrajectoryInOrderMatch     — expected tools appear in order, extras ok
  Builtin.TrajectoryAnyOrderMatch    — all expected tools present, any order

Verified against `strands-agents-evals` v1.0.2: the SDK ships these as scorer
*functions* (`exact_match_scorer` / `in_order_match_scorer` /
`any_order_match_scorer` in `trajectory_evaluator`) that the LLM
`TrajectoryEvaluator` calls internally. They are NOT standalone deterministic
evaluators. SAES wraps them directly as native `Evaluator` subclasses so they
run without an LLM — deterministic, cheap, exactly matching AgentCore's intent.

`@tool`-decorated scorers are called via `.__wrapped__` to get the plain
function (bypassing the tool-invocation wrapper).
"""

from __future__ import annotations

from typing import Any

from strands_evals.evaluators import Evaluator
from strands_evals.evaluators.trajectory_evaluator import (
    any_order_match_scorer,
    exact_match_scorer,
    in_order_match_scorer,
)
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput

# unwrap the @tool decorator to get the plain scoring functions
_SCORERS = {
    "Builtin.TrajectoryExactOrderMatch": exact_match_scorer.__wrapped__,
    "Builtin.TrajectoryInOrderMatch": in_order_match_scorer.__wrapped__,
    "Builtin.TrajectoryAnyOrderMatch": any_order_match_scorer.__wrapped__,
}

TRAJECTORY_EVALUATOR_IDS = frozenset(_SCORERS)


def _tool_names_from_session(session: Any) -> list[str]:
    """Extract the ordered tool-call names from a native Session.

    Reads `ToolExecutionSpan.tool_call.name` across all traces/spans. Tolerant
    of span shapes that don't expose a tool_call (skipped)."""
    names: list[str] = []
    for trace in getattr(session, "traces", []) or []:
        for span in getattr(trace, "spans", []) or []:
            tool_call = getattr(span, "tool_call", None)
            name = getattr(tool_call, "name", None) if tool_call else None
            if name:
                names.append(name)
    return names


def _actual_trajectory(case: EvaluationData) -> list[str]:
    """Actual tool-name sequence: from the Session if present, else from a
    pre-extracted list on actual_trajectory."""
    traj = case.actual_trajectory
    if traj is None:
        return []
    # native Session -> extract names; fall back to SAES-supplemented names
    # (attached by the CloudWatch task for non-Strands agents whose tool spans
    # the native mapper doesn't reconstruct — SPEC F6 fix).
    if hasattr(traj, "traces"):
        names = _tool_names_from_session(traj)
        if not names:
            supplemented = getattr(traj, "_saes_tool_names", None)
            if supplemented:
                return list(supplemented)
        return names
    # already a list of names (or {name: ...} dicts)
    names: list[str] = []
    for item in traj:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("name"):
            names.append(item["name"])
        elif getattr(item, "name", None):
            names.append(item.name)
    return names


class TrajectoryMatchEvaluator(Evaluator):
    """Deterministic trajectory scorer wrapping a native matcher function."""

    def __init__(self, evaluator_id: str, name: str | None = None):
        super().__init__(name=name or evaluator_id)
        if evaluator_id not in _SCORERS:
            raise KeyError(f"unknown trajectory evaluator '{evaluator_id}'")
        self._id = evaluator_id
        self._scorer = _SCORERS[evaluator_id]

    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        expected = list(evaluation_case.expected_trajectory or [])
        actual = _actual_trajectory(evaluation_case)

        if not expected:
            # no ground-truth trajectory -> not applicable, report cleanly
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason="no expected_trajectory provided (ground-truth required)",
                    label="N/A",
                )
            ]

        score = float(self._scorer(actual, expected))
        return [
            EvaluationOutput(
                score=score,
                test_pass=score >= 1.0,
                reason=(
                    f"{self._id}: actual={actual} vs expected={expected} -> {score:.3f}"
                ),
                label=f"{score:.2f}",
            )
        ]

    async def evaluate_async(
        self, evaluation_case: EvaluationData
    ) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)


def build_trajectory_evaluator(evaluator_id: str, name: str | None = None) -> Evaluator:
    return TrajectoryMatchEvaluator(evaluator_id, name=name)


__all__ = [
    "TRAJECTORY_EVALUATOR_IDS",
    "TrajectoryMatchEvaluator",
    "build_trajectory_evaluator",
]
