"""T2 judge layer tests — structured output parsing + repair/retry loop.

The parsing tests need no network. The retry-loop tests use a stub Strands
Model injected into StrandsJudge, so no real endpoint is contacted.
"""

import pytest

from saes.config.schema import JudgeModelConfig, JudgeProvider
from saes.judge import (
    ParseError,
    StrandsJudge,
    Verdict,
    extract_json,
    parse_verdict_fields,
)

# ---- structured-output extraction -------------------------------------------

def test_extract_bare_json():
    obj = extract_json('{"reasoning": "ok", "score": "Completely Yes"}')
    assert obj["score"] == "Completely Yes"


def test_extract_fenced_json():
    text = 'Here is my verdict:\n```json\n{"reasoning": "r", "score": "Yes"}\n```'
    obj = extract_json(text)
    assert obj["reasoning"] == "r"


def test_extract_embedded_json_in_prose():
    text = 'The answer is {"reasoning": "because", "verdict": "CORRECT"} — done.'
    obj = extract_json(text)
    assert obj["verdict"] == "CORRECT"


def test_extract_handles_braces_in_strings():
    text = '{"reasoning": "uses {curly} braces", "score": "No"}'
    obj = extract_json(text)
    assert obj["reasoning"] == "uses {curly} braces"


def test_extract_empty_raises():
    with pytest.raises(ParseError):
        extract_json("   ")


def test_extract_no_json_raises():
    with pytest.raises(ParseError):
        extract_json("I refuse to answer in JSON.")


def test_parse_verdict_field_aliases():
    reason, score = parse_verdict_fields({"reasoning": "r", "score": "Yes"})
    assert (reason, score) == ("r", "Yes")
    reason, score = parse_verdict_fields({"reason": "r2", "verdict": "SUCCESS"})
    assert (reason, score) == ("r2", "SUCCESS")


def test_parse_verdict_missing_score_raises():
    with pytest.raises(ParseError, match="missing a score"):
        parse_verdict_fields({"reasoning": "only reason"})


# ---- retry / repair loop with a stub model ----------------------------------

class _StubResult:
    def __init__(self, message: str):
        self.message = message
        self.metrics = None


class _StubAgentModel:
    """Stands in for a Strands Model; StrandsJudge wraps it in an Agent, but we
    monkeypatch Agent to route to this instead (see fixtures below)."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.calls: list[str] = []

    def next(self, prompt: str) -> str:
        self.calls.append(prompt)
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


@pytest.fixture
def patch_agent(monkeypatch):
    """Patch strands.Agent so StrandsJudge.score() drives our stub model."""

    def _install(stub: _StubAgentModel):
        class _FakeAgent:
            def __init__(self, model=None, callback_handler=None):
                self._stub = model

            async def invoke_async(self, prompt: str):
                return _StubResult(self._stub.next(prompt))

        import strands

        monkeypatch.setattr(strands, "Agent", _FakeAgent)
        return stub

    return _install


def _cfg(**kw) -> JudgeModelConfig:
    base = dict(
        provider=JudgeProvider.OPENAI_COMPATIBLE,
        model="test-judge",
        base_url="https://stub/v1",
        max_retries=2,
    )
    base.update(kw)
    return JudgeModelConfig(**base)


async def test_score_parses_clean_verdict(patch_agent):
    stub = patch_agent(
        _StubAgentModel(['{"reasoning": "solid", "score": "Completely Yes"}'])
    )
    judge = StrandsJudge(_cfg(), model=stub)
    verdict = await judge.score("rate this", schema={})
    assert isinstance(verdict, Verdict)
    assert verdict.errored is False
    assert verdict.raw_score == "Completely Yes"
    assert verdict.reason == "solid"
    assert verdict.judge_model == "test-judge"
    assert len(stub.calls) == 1


async def test_score_repairs_then_succeeds(patch_agent):
    stub = patch_agent(
        _StubAgentModel(
            [
                "no json here, sorry",  # attempt 1: unparseable
                '{"reasoning": "ok now", "score": "Yes"}',  # repair attempt succeeds
            ]
        )
    )
    judge = StrandsJudge(_cfg(max_retries=2), model=stub)
    verdict = await judge.score("rate this", schema={})
    assert verdict.errored is False
    assert verdict.raw_score == "Yes"
    assert len(stub.calls) == 2
    # the repair prompt should differ from the original
    assert stub.calls[1] != stub.calls[0]
    assert "could not be parsed" in stub.calls[1]


async def test_score_errors_after_exhausting_retries(patch_agent):
    stub = patch_agent(_StubAgentModel(["nope", "still nope", "nope again"]))
    judge = StrandsJudge(_cfg(max_retries=2), model=stub)
    verdict = await judge.score("rate this", schema={})
    assert verdict.errored is True
    assert verdict.label == "ERROR"
    assert verdict.raw_response is not None
    # max_retries=2 -> 3 attempts total
    assert len(stub.calls) == 3


async def test_score_survives_invocation_exception(patch_agent, monkeypatch):
    """If the model raises, it counts as a failed attempt, not a crash."""

    class _Boom(_StubAgentModel):
        def next(self, prompt: str) -> str:
            self.calls.append(prompt)
            if len(self.calls) == 1:
                raise RuntimeError("endpoint down")
            return '{"reasoning": "recovered", "score": "No"}'

    stub = patch_agent(_Boom([""]))
    judge = StrandsJudge(_cfg(max_retries=2), model=stub)
    verdict = await judge.score("rate this", schema={})
    assert verdict.errored is False
    assert verdict.raw_score == "No"
    assert len(stub.calls) == 2
