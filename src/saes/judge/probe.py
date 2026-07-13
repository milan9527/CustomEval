"""Judge capability probe (SPEC §3.5).

The native evaluators score via `Agent.invoke_async(prompt,
structured_output_model=...)`, so a judge endpoint MUST support tool calling /
structured output. A text-only endpoint fails every evaluation with
`StructuredOutputException`. This probe issues one tiny structured-output
request through the resolved judge model and reports whether it qualifies —
an explicit preflight so runs don't fail opaquely mid-flight.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..config.schema import JudgeModelConfig
from .providers import build_model


class _ProbeSchema(BaseModel):
    """Trivial structured-output target for the capability check."""

    ok: bool


@dataclass
class ProbeResult:
    supported: bool
    detail: str
    model: str


async def probe_judge_async(cfg: JudgeModelConfig, model: Any | None = None) -> ProbeResult:
    """Check whether the configured judge supports structured output.

    Returns a ProbeResult; never raises for an unsupported endpoint (that's a
    normal negative result), only for genuinely unexpected errors it re-raises
    after wrapping context.
    """
    from strands import Agent
    from strands.types.exceptions import StructuredOutputException

    judge_model = model if model is not None else build_model(cfg)
    agent = Agent(model=judge_model, callback_handler=None)
    prompt = "Return a structured result with ok=true. This is a capability check."

    try:
        result = await agent.invoke_async(prompt, structured_output_model=_ProbeSchema)
    except StructuredOutputException as exc:
        return ProbeResult(
            supported=False,
            detail=(
                "endpoint did not produce structured output via tool calling "
                f"({exc}). The judge must support tool calling / structured "
                "output; a text-only chat-completions endpoint is not supported."
            ),
            model=cfg.model,
        )
    except Exception as exc:  # noqa: BLE001 - connection/auth/etc.
        return ProbeResult(
            supported=False,
            detail=f"probe call failed ({type(exc).__name__}: {exc})",
            model=cfg.model,
        )

    structured = getattr(result, "structured_output", None)
    if structured is not None:
        return ProbeResult(
            supported=True,
            detail="structured output confirmed via tool calling",
            model=cfg.model,
        )
    return ProbeResult(
        supported=False,
        detail="no structured_output returned by the judge",
        model=cfg.model,
    )


def probe_judge(cfg: JudgeModelConfig, model: Any | None = None) -> ProbeResult:
    """Synchronous wrapper for CLI use."""
    return asyncio.run(probe_judge_async(cfg, model=model))


__all__ = ["ProbeResult", "probe_judge", "probe_judge_async"]
