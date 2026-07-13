"""SAES EvaluationResult (SPEC §2.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """A single evaluator's verdict on one unit (session/trace/span)."""

    evaluator_id: str
    level: str
    session_id: str
    score: float
    reason: str
    trace_id: str | None = None
    span_id: str | None = None
    label: str | None = None
    raw_score: Any = None
    judge_model: str | None = None
    higher_is_better: bool = True
    ground_truth_used: bool = False
    ignored_reference_input_fields: list[str] = field(default_factory=list)
    template_source: str = ""
    template_version: str = ""
    errored: bool = False
    error_detail: str | None = None
    latency_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["EvaluationResult"]
