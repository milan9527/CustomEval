"""T13 — deterministic trajectory ground-truth scorers (SPEC §4.2, §5).

Wraps the native scorer functions (exact/in-order/any-order); no LLM. Verified
against the real matcher math from strands_evals.
"""

import pytest
from strands_evals.types.evaluation import EvaluationData

from saes.config.schema import EvaluatorRef
from saes.evaluators import (
    TRAJECTORY_EVALUATOR_IDS,
    available_builtins,
    resolve_evaluator,
)
from saes.evaluators.trajectory import TrajectoryMatchEvaluator

_MODEL = object()  # trajectory scorers ignore the judge model (deterministic)


def _case(actual, expected):
    return EvaluationData(
        input="q", actual_trajectory=list(actual), expected_trajectory=list(expected)
    )


def test_three_trajectory_ids_registered():
    assert TRAJECTORY_EVALUATOR_IDS == {
        "Builtin.TrajectoryExactOrderMatch",
        "Builtin.TrajectoryInOrderMatch",
        "Builtin.TrajectoryAnyOrderMatch",
    }
    for tid in TRAJECTORY_EVALUATOR_IDS:
        assert tid in available_builtins()


def test_resolve_returns_deterministic_evaluator():
    ev = resolve_evaluator(
        EvaluatorRef(id="Builtin.TrajectoryExactOrderMatch", type="builtin"), _MODEL
    )
    assert isinstance(ev, TrajectoryMatchEvaluator)
    assert ev.get_name() == "Builtin.TrajectoryExactOrderMatch"


def test_exact_order_match():
    ev = TrajectoryMatchEvaluator("Builtin.TrajectoryExactOrderMatch")
    assert ev.evaluate(_case(["a", "b"], ["a", "b"]))[0].score == 1.0
    # one wrong position -> 0.5 (native exact_match math)
    assert ev.evaluate(_case(["a", "x"], ["a", "b"]))[0].score == 0.5
    out = ev.evaluate(_case(["a", "b"], ["a", "b"]))[0]
    assert out.test_pass is True


def test_in_order_match_allows_extras():
    ev = TrajectoryMatchEvaluator("Builtin.TrajectoryInOrderMatch")
    # extras between expected tools, still in order -> 1.0
    assert ev.evaluate(_case(["a", "z", "b"], ["a", "b"]))[0].score == 1.0


def test_any_order_match():
    ev = TrajectoryMatchEvaluator("Builtin.TrajectoryAnyOrderMatch")
    # reversed order still matches for any-order -> 1.0
    assert ev.evaluate(_case(["b", "a"], ["a", "b"]))[0].score == 1.0


def test_no_expected_trajectory_is_not_applicable():
    ev = TrajectoryMatchEvaluator("Builtin.TrajectoryExactOrderMatch")
    out = ev.evaluate(_case(["a"], []))[0]
    assert out.label == "N/A"
    assert out.test_pass is False
    assert "ground-truth required" in out.reason


def test_extracts_tool_names_from_session():
    """actual_trajectory as a native Session -> tool names pulled from spans."""

    class _ToolCall:
        def __init__(self, name):
            self.name = name

    class _Span:
        def __init__(self, name):
            self.tool_call = _ToolCall(name)

    class _Trace:
        def __init__(self, names):
            self.spans = [_Span(n) for n in names]

    class _Session:
        def __init__(self, names):
            self.traces = [_Trace(names)]

    case = EvaluationData(input="q", expected_trajectory=["search", "book"])
    case.actual_trajectory = _Session(["search", "book"])
    ev = TrajectoryMatchEvaluator("Builtin.TrajectoryExactOrderMatch")
    out = ev.evaluate(case)[0]
    assert out.score == 1.0


def test_supplemented_tool_names_fallback():
    """When the Session has no ToolExecutionSpans (non-Strands agent) but the
    CloudWatch task attached `_saes_tool_names` (F6 supplement), the evaluator
    uses those names."""

    class _Session:
        def __init__(self):
            self.traces = []          # no tool spans
            self._saes_tool_names = ["get_weather", "calculate"]

    case = EvaluationData(input="q", expected_trajectory=["get_weather", "calculate"])
    case.actual_trajectory = _Session()
    ev = TrajectoryMatchEvaluator("Builtin.TrajectoryInOrderMatch")
    assert ev.evaluate(case)[0].score == 1.0


def test_real_session_spans_take_precedence_over_supplement():
    """If the Session DOES have tool spans, use them (don't override with the
    supplement)."""

    class _TC:
        def __init__(self, n): self.name = n
    class _Span:
        def __init__(self, n): self.tool_call = _TC(n)
    class _Trace:
        def __init__(self, ns): self.spans = [_Span(n) for n in ns]
    class _Session:
        def __init__(self):
            self.traces = [_Trace(["search"])]
            self._saes_tool_names = ["wrong", "names"]  # must be ignored

    case = EvaluationData(input="q", expected_trajectory=["search"])
    case.actual_trajectory = _Session()
    ev = TrajectoryMatchEvaluator("Builtin.TrajectoryExactOrderMatch")
    assert ev.evaluate(case)[0].score == 1.0


def test_unknown_trajectory_id_raises():
    with pytest.raises(KeyError):
        TrajectoryMatchEvaluator("Builtin.TrajectoryNope")
