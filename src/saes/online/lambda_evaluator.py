"""Custom code evaluator as an AWS Lambda (SPEC §6.2, T16).

AgentCore's code-based evaluators are Lambda functions: AgentCore passes the
agent's data as a structured event and expects `{score, label, reason}` back.
SAES makes the *same* `@code_evaluator` function body work both locally (via
`CodeEvaluator`) and as a Lambda — you deploy this handler, and it dispatches
to the registered function by id.

Event contract (tolerant):
    { "evaluator_id": "paystub_amount",
      "input": <task input>, "actual_output": <agent response>,
      "expected_output": ..., "actual_trajectory": ..., "metadata": {...} }
Response:
    { "score": float, "label": str|None, "reason": str, "test_pass": bool }

The event maps onto a native `EvaluationData`; the function returns a
`CodeVerdict`; the handler serializes it to the AgentCore-style response.
"""

from __future__ import annotations

from typing import Any

from strands_evals.types.evaluation import EvaluationData

from ..evaluators.custom import _CODE_EVALUATORS, _to_output


def _event_to_evaluation_data(event: dict[str, Any]) -> EvaluationData:
    """Map a Lambda event to a native EvaluationData.

    Only fields present are set; a code evaluator reads what it needs.
    """
    return EvaluationData(
        input=event.get("input"),
        actual_output=event.get("actual_output") or event.get("output"),
        expected_output=event.get("expected_output"),
        actual_trajectory=event.get("actual_trajectory"),
        expected_trajectory=event.get("expected_trajectory"),
        metadata=event.get("metadata"),
    )


def handle(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Lambda entry point. Dispatches to a registered code evaluator by id.

    `evaluator_id` selects the function; it must have been registered via
    `@code_evaluator(id=...)` in code imported by the Lambda package.
    """
    evaluator_id = event.get("evaluator_id")
    if not evaluator_id:
        return _error("event missing 'evaluator_id'")

    entry = _CODE_EVALUATORS.get(evaluator_id)
    if entry is None:
        return _error(
            f"code evaluator '{evaluator_id}' is not registered in this Lambda; "
            "ensure the module defining it is imported at package load."
        )

    fn, _level = entry
    try:
        verdict = fn(_event_to_evaluation_data(event))
    except Exception as exc:  # noqa: BLE001 - report as a failed evaluation, don't 500
        return _error(f"evaluator raised: {type(exc).__name__}: {exc}")

    out = _to_output(verdict)
    return {
        "score": out.score,
        "label": out.label,
        "reason": out.reason,
        "test_pass": out.test_pass,
    }


def _error(message: str) -> dict[str, Any]:
    return {"score": 0.0, "label": "ERROR", "reason": message, "test_pass": False}


__all__ = ["handle"]
