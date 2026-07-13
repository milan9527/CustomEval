"""SAES evaluators (SPEC §4, §6).

Built-ins map to NATIVE strands_evals evaluators (no reimplementation).
Custom LLM and custom code evaluators provide AgentCore-equivalent
extensibility on top of the same native primitives.
"""

from .custom import (
    CodeEvaluator,
    CodeVerdict,
    build_code_evaluator,
    build_custom_llm,
    code_evaluator,
)
from .registry import (
    BUILTIN_EVALUATORS,
    available_builtins,
    resolve_evaluator,
)
from .trajectory import (
    TRAJECTORY_EVALUATOR_IDS,
    TrajectoryMatchEvaluator,
    build_trajectory_evaluator,
)

__all__ = [
    "BUILTIN_EVALUATORS",
    "TRAJECTORY_EVALUATOR_IDS",
    "CodeEvaluator",
    "CodeVerdict",
    "TrajectoryMatchEvaluator",
    "available_builtins",
    "build_code_evaluator",
    "build_custom_llm",
    "build_trajectory_evaluator",
    "code_evaluator",
    "resolve_evaluator",
]
