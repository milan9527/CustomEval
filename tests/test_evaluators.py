"""Evaluator resolution tests — native builtins + custom LLM/code (SPEC §4, §6).

These assert that SAES correctly ASSEMBLES native strands_evals evaluators and
injects the selected judge — SAES does not reimplement evaluators. A sentinel
model object stands in for a real judge model (no network).
"""

import pytest
from strands_evals.evaluators import (
    CorrectnessEvaluator,
    Evaluator,
    HelpfulnessEvaluator,
    OutputEvaluator,
    ToolSelectionAccuracyEvaluator,
)

from saes.config.schema import EvaluatorRef
from saes.evaluators import (
    BUILTIN_EVALUATORS,
    CodeEvaluator,
    CodeVerdict,
    available_builtins,
    code_evaluator,
    resolve_evaluator,
)
from saes.evaluators.registry import CONTEXT_RELEVANCE_IS_ALIASED

_MODEL = object()  # sentinel judge model; never invoked in these tests


# ---- built-ins map to NATIVE evaluators -------------------------------------

def test_all_13_builtins_registered():
    assert len(BUILTIN_EVALUATORS) == 14  # 13 ids + ContextRelevance alias
    assert "Builtin.Helpfulness" in BUILTIN_EVALUATORS
    assert "Builtin.ToolParameterAccuracy" in BUILTIN_EVALUATORS


def test_resolve_builtin_returns_native_instance():
    ref = EvaluatorRef(id="Builtin.Helpfulness", type="builtin")
    ev = resolve_evaluator(ref, _MODEL)
    assert isinstance(ev, HelpfulnessEvaluator)
    assert isinstance(ev, Evaluator)  # native base class


def test_resolve_builtin_injects_judge_model():
    ref = EvaluatorRef(id="Builtin.Correctness", type="builtin")
    ev = resolve_evaluator(ref, _MODEL)
    assert isinstance(ev, CorrectnessEvaluator)
    # the injected model is the SAES-selected judge
    assert getattr(ev, "model", None) is _MODEL or _MODEL in vars(ev).values()


def test_resolve_tool_evaluator():
    ref = EvaluatorRef(id="Builtin.ToolSelectionAccuracy", type="builtin")
    ev = resolve_evaluator(ref, _MODEL)
    assert isinstance(ev, ToolSelectionAccuracyEvaluator)


def test_builtin_named_by_id_for_unique_experiment_names():
    # native Experiment rejects duplicate evaluator names; SAES names each by
    # its id so the same class can appear twice (e.g. different judge override).
    ev = resolve_evaluator(EvaluatorRef(id="Builtin.Helpfulness", type="builtin"), _MODEL)
    assert ev.get_name() == "Builtin.Helpfulness"


def test_context_relevance_is_aliased_flag():
    # documented, not silent (SPEC §4 note)
    assert CONTEXT_RELEVANCE_IS_ALIASED is True
    ref = EvaluatorRef(id="Builtin.ContextRelevance", type="builtin")
    ev = resolve_evaluator(ref, _MODEL)
    assert isinstance(ev, Evaluator)


def test_unknown_builtin_raises():
    ref = EvaluatorRef(id="Builtin.Nope", type="builtin")
    with pytest.raises(KeyError, match="unknown built-in evaluator"):
        resolve_evaluator(ref, _MODEL)


def test_available_builtins_sorted():
    ids = available_builtins()
    assert ids == sorted(ids)
    assert "Builtin.GoalSuccessRate" in ids


# ---- custom LLM evaluator (SPEC §6.1) ---------------------------------------

def test_custom_llm_wraps_output_evaluator():
    ref = EvaluatorRef(
        id="hipaa_compliance",
        type="llm",
        level="trace",
        instructions="Score 1.0 if no PHI disclosed, else 0.0.",
    )
    ev = resolve_evaluator(ref, _MODEL)
    assert isinstance(ev, OutputEvaluator)


def test_custom_llm_requires_instructions():
    ref = EvaluatorRef(id="bad", type="llm")
    with pytest.raises(ValueError, match="requires `instructions`"):
        resolve_evaluator(ref, _MODEL)


# ---- custom code evaluator (SPEC §6.2) --------------------------------------

def test_code_evaluator_registration_and_resolution():
    @code_evaluator(id="paystub_amount", level="trace")
    def check(case) -> CodeVerdict:
        ok = "$8,333.33" in str(case.actual_output)
        return CodeVerdict(
            score=1.0 if ok else 0.0,
            label="PASS" if ok else "FAIL",
            reason="verbatim amount present" if ok else "amount missing",
        )

    ref = EvaluatorRef(id="paystub_amount", type="code")
    ev = resolve_evaluator(ref, _MODEL)  # model ignored for code evaluators
    assert isinstance(ev, CodeEvaluator)


def test_code_evaluator_runs_deterministically():
    @code_evaluator(id="contains_ok")
    def check(case) -> CodeVerdict:
        ok = "ok" in str(case.actual_output).lower()
        return CodeVerdict(score=1.0 if ok else 0.0, label="PASS" if ok else "FAIL")

    ref = EvaluatorRef(id="contains_ok", type="code")
    ev = resolve_evaluator(ref, _MODEL)

    from strands_evals.types.evaluation import EvaluationData

    good = EvaluationData(input="q", actual_output="all OK here")
    bad = EvaluationData(input="q", actual_output="nope")
    out_good = ev.evaluate(good)
    out_bad = ev.evaluate(bad)
    assert out_good[0].score == 1.0
    assert out_good[0].test_pass is True
    assert out_bad[0].score == 0.0
    assert out_bad[0].test_pass is False


def test_unregistered_code_evaluator_raises():
    ref = EvaluatorRef(id="never_registered", type="code")
    with pytest.raises(KeyError, match="not registered"):
        resolve_evaluator(ref, _MODEL)
