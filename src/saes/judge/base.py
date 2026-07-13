"""Judge verdict types and the Judge protocol (SPEC §3.3, T2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Verdict:
    """A judge's structured decision (SPEC §3.3).

    `reason` precedes `score` by convention — the judge reasons before scoring.
    `raw_score` preserves the verdict as the judge emitted it (e.g. an ordinal
    label like "Completely Yes") before normalization to `score` in [0,1].
    """

    reason: str
    score: float
    label: str | None = None
    raw_score: Any = None
    judge_model: str | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    errored: bool = False
    raw_response: str | None = None

    @classmethod
    def error(cls, message: str, raw_response: str | None, judge_model: str | None) -> Verdict:
        return cls(
            reason=message,
            score=0.0,
            label="ERROR",
            errored=True,
            raw_response=raw_response,
            judge_model=judge_model,
        )


@runtime_checkable
class Judge(Protocol):
    """A resolved judge model that can score a rendered prompt against a schema."""

    model_id: str

    async def score(self, prompt: str, schema: dict[str, Any]) -> Verdict: ...


class JudgeError(RuntimeError):
    """Raised for unrecoverable judge failures (after retries are exhausted)."""


__all__ = ["Judge", "JudgeError", "TokenUsage", "Verdict"]
