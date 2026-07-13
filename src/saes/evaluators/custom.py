"""Custom evaluators (SPEC §6) — parity with AgentCore custom evaluators.

Two mechanisms, both built on native strands_evals primitives:

1. Custom LLM evaluator  -> wraps native `OutputEvaluator(rubric=..., model=judge)`.
   The user supplies a rubric/instructions; SAES injects the selected judge.
   Mirrors AgentCore's custom LLM-as-a-Judge evaluators.

2. Custom code evaluator -> a deterministic Python callable adapted to the
   native `Evaluator` interface. Mirrors AgentCore's Lambda code-based
   evaluators (SPEC §6.2): no language understanding, exact/format/business
   checks. The callable receives an EvaluationData view and returns a Verdict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from strands_evals.evaluators import Evaluator, OutputEvaluator
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput

from ..config.schema import EvaluatorRef


def build_custom_llm(ref: EvaluatorRef, judge_model: Any) -> Evaluator:
    """Build a custom LLM evaluator from a config ref.

    `ref.instructions` is the rubric. The SAES judge is injected as the model,
    so custom evaluators use the same OpenAI-compatible judge selection as
    built-ins (SPEC §3)."""
    if not ref.instructions:
        raise ValueError(
            f"custom LLM evaluator '{ref.id}' requires `instructions` (the rubric)"
        )
    return OutputEvaluator(
        rubric=ref.instructions,
        model=judge_model,
        name=ref.id,
    )


# --- custom code evaluators --------------------------------------------------


@dataclass
class CodeVerdict:
    """Return type for a custom code evaluator function."""

    score: float
    label: str | None = None
    reason: str = ""
    test_pass: bool | None = None


# A code evaluator function maps an EvaluationData to a CodeVerdict.
CodeEvalFn = Callable[[EvaluationData], CodeVerdict]


class CodeEvaluator(Evaluator):
    """Adapts a deterministic Python callable to the native Evaluator interface.

    This is the local/in-process form. The online worker (M3) deploys the same
    callable as a Lambda; the function body is unchanged (SPEC §6.2)."""

    def __init__(self, fn: CodeEvalFn, evaluator_id: str, level: str = "trace"):
        super().__init__()
        self._fn = fn
        self._id = evaluator_id
        self._level = level

    def get_name(self) -> str:  # pragma: no cover - trivial
        return self._id

    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        verdict = self._fn(evaluation_case)
        return [_to_output(verdict)]

    async def evaluate_async(
        self, evaluation_case: EvaluationData
    ) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)


def _to_output(v: CodeVerdict) -> EvaluationOutput:
    test_pass = v.test_pass if v.test_pass is not None else v.score >= 0.5
    return EvaluationOutput(
        score=float(v.score),
        test_pass=bool(test_pass),
        reason=v.reason,
        label=v.label,
    )


# Registry for code evaluators registered via the decorator.
_CODE_EVALUATORS: dict[str, tuple[CodeEvalFn, str]] = {}


def code_evaluator(id: str, level: str = "trace") -> Callable[[CodeEvalFn], CodeEvalFn]:
    """Decorator to register a deterministic code evaluator (SPEC §6.2).

    Example:
        @code_evaluator(id="paystub_amount", level="trace")
        def check(case) -> CodeVerdict:
            ok = "$8,333.33" in str(case.actual_output)
            return CodeVerdict(1.0 if ok else 0.0, "PASS" if ok else "FAIL")
    """

    def _decorator(fn: CodeEvalFn) -> CodeEvalFn:
        _CODE_EVALUATORS[id] = (fn, level)
        return fn

    return _decorator


def build_code_evaluator(ref: EvaluatorRef) -> Evaluator:
    """Resolve a registered code evaluator id to a native Evaluator."""
    entry = _CODE_EVALUATORS.get(ref.id)
    if entry is None:
        raise KeyError(
            f"code evaluator '{ref.id}' is not registered; "
            "register it with @code_evaluator(id=...) before running."
        )
    fn, level = entry
    return CodeEvaluator(fn, evaluator_id=ref.id, level=ref.level or level)


__all__ = [
    "CodeEvaluator",
    "CodeVerdict",
    "build_code_evaluator",
    "build_custom_llm",
    "code_evaluator",
]
