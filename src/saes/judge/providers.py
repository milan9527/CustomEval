"""Judge model resolution over Strands model providers (SPEC §3.2).

`build_model()` turns a JudgeModelConfig into a native Strands Model — this is
what gets injected into the native strands_evals evaluators (built-in and
custom). The differentiator is `openai_compatible`, which routes any endpoint
exposing /v1/chat/completions (OpenAI, Azure, vLLM, Ollama, LiteLLM,
SageMaker, ...) through Strands' OpenAIModel. `bedrock` is also supported.

`resolve_judge()` additionally wraps a model as a `Judge` (structured verdict +
retry/repair loop) for any direct-judge use outside the evaluator pipeline.
"""

from __future__ import annotations

from typing import Any

from ..config.schema import JudgeModelConfig, JudgeProvider
from .base import Judge, TokenUsage, Verdict
from .structured import ParseError, extract_json, parse_verdict_fields


def build_model(cfg: JudgeModelConfig) -> Any:
    """Instantiate the native Strands Model for the configured judge provider.

    Imports are local so the config layer and tests that stub the judge don't
    pay the import cost or require the optional provider extras.
    """
    if cfg.provider is JudgeProvider.OPENAI_COMPATIBLE:
        from strands.models.openai import OpenAIModel

        client_args: dict[str, Any] = {"base_url": cfg.base_url}
        api_key = cfg.resolved_api_key()
        if api_key is not None:
            client_args["api_key"] = api_key
        return OpenAIModel(
            client_args=client_args,
            model_id=cfg.model,
            params=cfg.params or None,
        )

    if cfg.provider is JudgeProvider.BEDROCK:
        from strands.models import BedrockModel

        return BedrockModel(model_id=cfg.model, **(cfg.params or {}))

    if cfg.provider is JudgeProvider.STRANDS:
        raise NotImplementedError(
            "provider 'strands' (named provider registry) is not wired in M1; "
            "use 'openai_compatible' or 'bedrock'."
        )

    raise ValueError(f"unknown judge provider: {cfg.provider}")


class StrandsJudge:
    """A Judge backed by a Strands Model, with a parse/repair/retry loop.

    Uses a Strands Agent as the invocation surface so any Model provider works
    uniformly. Structured output is requested via the prompt/schema; parsing is
    tolerant (fenced JSON, embedded JSON) with one repair re-ask before the
    remaining retries, then an errored Verdict (never a silent drop).
    """

    def __init__(self, cfg: JudgeModelConfig, model: Any | None = None):
        self.cfg = cfg
        self.model_id = cfg.model
        self._model = model if model is not None else build_model(cfg)

    async def score(self, prompt: str, schema: dict[str, Any]) -> Verdict:
        from strands import Agent

        agent = Agent(model=self._model, callback_handler=None)
        attempts = self.cfg.max_retries + 1
        last_raw: str | None = None

        for attempt in range(attempts):
            ask = prompt if attempt == 0 else _repair_prompt(prompt, last_raw)
            try:
                result = await agent.invoke_async(ask)
            except Exception as exc:  # noqa: BLE001 - surface as errored verdict
                last_raw = f"invocation error: {exc}"
                continue

            text = _result_text(result)
            last_raw = text
            try:
                obj = extract_json(text)
                reason, raw_score = parse_verdict_fields(obj)
            except ParseError:
                continue

            return Verdict(
                reason=reason,
                score=0.0,  # normalized by the evaluator's scale map (T4)
                raw_score=raw_score,
                judge_model=self.model_id,
                token_usage=_token_usage(result),
                raw_response=text,
            )

        return Verdict.error(
            f"failed to parse a verdict after {attempts} attempt(s)",
            raw_response=last_raw,
            judge_model=self.model_id,
        )


def _repair_prompt(original: str, last_raw: str | None) -> str:
    return (
        original
        + "\n\nYour previous response could not be parsed as JSON:\n"
        + f"---\n{(last_raw or '')[:1000]}\n---\n"
        + "Return ONLY a single valid JSON object with the required fields, "
        "no prose, no code fences."
    )


def _result_text(result: Any) -> str:
    """Extract the text of a Strands AgentResult across minor API shapes."""
    for attr in ("message", "output", "content"):
        val = getattr(result, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    return str(result)


def _token_usage(result: Any) -> TokenUsage:
    metrics = getattr(result, "metrics", None)
    usage = getattr(metrics, "accumulated_usage", None) if metrics else None
    if isinstance(usage, dict):
        return TokenUsage(
            input_tokens=int(usage.get("inputTokens", 0) or 0),
            output_tokens=int(usage.get("outputTokens", 0) or 0),
        )
    return TokenUsage()


def resolve_judge(cfg: JudgeModelConfig, model: Any | None = None) -> Judge:
    """Resolve a JudgeModelConfig into a Judge instance.

    `model` lets callers/tests inject a pre-built or stub Strands Model.
    """
    return StrandsJudge(cfg, model=model)


__all__ = ["StrandsJudge", "build_model", "resolve_judge"]
