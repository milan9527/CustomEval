"""Judge capability probe tests (SPEC §3.5)."""

import pytest

from saes.config.schema import JudgeModelConfig
from saes.judge import probe_judge
from saes.judge.probe import _ProbeSchema, probe_judge_async


def _cfg():
    return JudgeModelConfig(
        provider="openai_compatible", model="test", base_url="https://x/v1"
    )


@pytest.fixture
def patch_agent(monkeypatch):
    """Patch strands.Agent so the probe drives a stub instead of a real model."""

    def _install(behavior):
        class _Result:
            def __init__(self, structured):
                self.structured_output = structured

        class _FakeAgent:
            def __init__(self, model=None, callback_handler=None):
                pass

            async def invoke_async(self, prompt, structured_output_model=None):
                return behavior(structured_output_model)

        import strands

        monkeypatch.setattr(strands, "Agent", _FakeAgent)

    return _install


async def test_probe_supported(patch_agent):
    patch_agent(lambda schema: type("R", (), {"structured_output": schema(ok=True)})())
    result = await probe_judge_async(_cfg(), model=object())
    assert result.supported is True
    assert "confirmed" in result.detail


async def test_probe_unsupported_structured_output_exception(patch_agent, monkeypatch):
    from strands.types.exceptions import StructuredOutputException

    def _raise(schema):
        raise StructuredOutputException("no tool call")

    patch_agent(_raise)
    result = await probe_judge_async(_cfg(), model=object())
    assert result.supported is False
    assert "text-only" in result.detail


async def test_probe_no_structured_output_returned(patch_agent):
    patch_agent(lambda schema: type("R", (), {"structured_output": None})())
    result = await probe_judge_async(_cfg(), model=object())
    assert result.supported is False
    assert "no structured_output" in result.detail


async def test_probe_connection_error(patch_agent):
    def _boom(schema):
        raise ConnectionError("refused")

    patch_agent(_boom)
    result = await probe_judge_async(_cfg(), model=object())
    assert result.supported is False
    assert "probe call failed" in result.detail


def test_probe_sync_wrapper(patch_agent):
    patch_agent(lambda schema: type("R", (), {"structured_output": schema(ok=True)})())
    result = probe_judge(_cfg(), model=object())
    assert result.supported is True


def test_probe_schema_is_trivial():
    # the probe target is a minimal structured type
    assert set(_ProbeSchema.model_fields) == {"ok"}
