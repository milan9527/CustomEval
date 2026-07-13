"""Evaluator registry (SPEC §4, §6).

Maps AgentCore-style evaluator ids (e.g. 'Builtin.Helpfulness') onto the
NATIVE `strands_evals` evaluators — SAES does not reimplement evaluators or
prompt templates. The only SAES-owned behavior here is:
  1. resolving a config `EvaluatorRef` to a native Evaluator instance, and
  2. injecting the SAES-selected judge model (any OpenAI-compatible endpoint).

Custom evaluators (SPEC §6) are resolved via build_custom_llm / the code
evaluator base in custom.py.
"""

from __future__ import annotations

from typing import Any

from strands_evals.evaluators import (
    CoherenceEvaluator,
    ConcisenessEvaluator,
    CorrectnessEvaluator,
    Evaluator,
    FaithfulnessEvaluator,
    GoalSuccessRateEvaluator,
    HarmfulnessEvaluator,
    HelpfulnessEvaluator,
    InstructionFollowingEvaluator,
    RefusalEvaluator,
    ResponseRelevanceEvaluator,
    StereotypingEvaluator,
    ToolParameterAccuracyEvaluator,
    ToolSelectionAccuracyEvaluator,
)

from ..config.schema import EvaluatorRef
from .custom import build_code_evaluator, build_custom_llm
from .trajectory import TRAJECTORY_EVALUATOR_IDS, build_trajectory_evaluator

# AgentCore built-in id -> native strands_evals evaluator class (SPEC §4.1).
# These are the 13 built-ins; native evaluators carry the AgentCore-equivalent
# prompt templates and versioned scoring, so scores line up with the SDK.
BUILTIN_EVALUATORS: dict[str, type[Evaluator]] = {
    "Builtin.GoalSuccessRate": GoalSuccessRateEvaluator,
    "Builtin.Helpfulness": HelpfulnessEvaluator,
    "Builtin.Correctness": CorrectnessEvaluator,
    "Builtin.Coherence": CoherenceEvaluator,
    "Builtin.Conciseness": ConcisenessEvaluator,
    "Builtin.Faithfulness": FaithfulnessEvaluator,
    "Builtin.Harmfulness": HarmfulnessEvaluator,
    "Builtin.InstructionFollowing": InstructionFollowingEvaluator,
    "Builtin.ResponseRelevance": ResponseRelevanceEvaluator,
    "Builtin.ContextRelevance": ResponseRelevanceEvaluator,  # see note below
    "Builtin.Refusal": RefusalEvaluator,
    "Builtin.Stereotyping": StereotypingEvaluator,
    "Builtin.ToolSelectionAccuracy": ToolSelectionAccuracyEvaluator,
    "Builtin.ToolParameterAccuracy": ToolParameterAccuracyEvaluator,
}
# NOTE: strands_evals v1.0.2 exposes no distinct ContextRelevanceEvaluator;
# it is mapped to ResponseRelevance as the closest native evaluator. Revisit
# when the SDK adds a dedicated one. (Surfaced, not silently aliased.)

CONTEXT_RELEVANCE_IS_ALIASED = True


def available_builtins() -> list[str]:
    return sorted(set(BUILTIN_EVALUATORS) | TRAJECTORY_EVALUATOR_IDS)


def resolve_evaluator(ref: EvaluatorRef, judge_model: Any) -> Evaluator:
    """Resolve one config EvaluatorRef to a native strands_evals Evaluator,
    injecting the SAES judge model where the evaluator uses an LLM judge.

    `judge_model` is a strands Model instance (or model-id str) — see
    saes.judge.providers.build_model.
    """
    if ref.type == "builtin":
        return _resolve_builtin(ref, judge_model)
    if ref.type == "llm":
        return build_custom_llm(ref, judge_model)
    if ref.type == "code":
        return build_code_evaluator(ref)
    raise ValueError(f"unknown evaluator type: {ref.type!r}")


def _resolve_builtin(ref: EvaluatorRef, judge_model: Any) -> Evaluator:
    # Trajectory matchers are deterministic (no LLM) — wrap native scorer funcs.
    if ref.id in TRAJECTORY_EVALUATOR_IDS:
        return build_trajectory_evaluator(ref.id, name=ref.id)

    cls = BUILTIN_EVALUATORS.get(ref.id)
    if cls is None:
        raise KeyError(
            f"unknown built-in evaluator '{ref.id}'; "
            f"available: {available_builtins()}"
        )
    # All native builtins accept `model=` and `name=`. The judge is injected
    # uniformly; `name=ref.id` keeps evaluator names unique within an
    # Experiment (the native engine rejects duplicate names — e.g. the same
    # class used twice with different judge overrides) and stamps results with
    # the AgentCore-style id.
    return cls(model=judge_model, name=ref.id)


__all__ = [
    "BUILTIN_EVALUATORS",
    "CONTEXT_RELEVANCE_IS_ALIASED",
    "available_builtins",
    "resolve_evaluator",
]
