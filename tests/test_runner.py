"""T5 on-demand runner tests.

The aggregation is pure and tested directly. The end-to-end run is tested with
a stub evaluator (native Evaluator subclass returning fixed EvaluationOutputs)
and stub sessions, so it exercises the real native Experiment pipeline without
a judge model or AWS.
"""

import pytest
from strands_evals.evaluators import Evaluator
from strands_evals.types.evaluation import EvaluationOutput

from saes.config.schema import DataSourceConfig, EvaluationConfig, JudgeModelConfig
from saes.run import run_on_demand
from saes.run.runner import _aggregate

# ---- aggregation (pure) -----------------------------------------------------

class _Out:
    def __init__(self, score, test_pass=True, label=None):
        self.score = score
        self.test_pass = test_pass
        self.label = label


class _Report:
    """Mimics the native report: flattened rows, one per (evaluator, case),
    with report.cases[i]['evaluator'] naming the evaluator (evaluator-major)."""

    def __init__(self, rows):
        # rows: list of (evaluator_id, EvaluationOutput)
        self.detailed_results = [[out] for _, out in rows]
        self.cases = [{"evaluator": ev} for ev, _ in rows]


def test_aggregate_avg_and_pass_rate():
    # two evaluators × two cases, evaluator-major flattening
    report = _Report(
        [
            ("Builtin.Helpfulness", _Out(1.0, True)),
            ("Builtin.Helpfulness", _Out(0.0, False)),
            ("Builtin.Correctness", _Out(0.5, False)),
            ("Builtin.Correctness", _Out(1.0, True)),
        ]
    )
    agg = _aggregate(report, ["Builtin.Helpfulness", "Builtin.Correctness"])
    assert agg["Builtin.Helpfulness"]["avg"] == 0.5
    assert agg["Builtin.Helpfulness"]["pass_rate"] == 0.5
    assert agg["Builtin.Correctness"]["avg"] == 0.75
    assert agg["Builtin.Correctness"]["n"] == 2.0


def test_aggregate_excludes_errored():
    report = _Report(
        [
            ("Builtin.Helpfulness", _Out(1.0, True)),
            ("Builtin.Helpfulness", _Out(0.0, False, label="ERROR")),
        ]
    )
    agg = _aggregate(report, ["Builtin.Helpfulness"])
    stats = agg["Builtin.Helpfulness"]
    assert stats["errored"] == 1.0
    assert stats["n"] == 1.0  # errored excluded from n
    assert stats["avg"] == 1.0  # only the non-errored score counts


def test_aggregate_empty():
    agg = _aggregate(_Report([]), ["Builtin.Helpfulness"])
    assert agg["Builtin.Helpfulness"] == {
        "avg": 0.0,
        "pass_rate": 0.0,
        "n": 0.0,
        "errored": 0.0,
    }


# ---- end-to-end run through the native Experiment ---------------------------

class _StubSession:
    def __init__(self, session_id, response):
        self.session_id = session_id
        self.final_response = response
        self.traces = []


class _FixedEvaluator(Evaluator):
    """Native Evaluator subclass that returns a fixed score — no judge."""

    def __init__(self, score, name=None):
        super().__init__(name=name)
        self._score = score

    def evaluate(self, evaluation_case):
        return [
            EvaluationOutput(
                score=self._score,
                test_pass=self._score >= 0.5,
                reason="stub",
                label="stub",
            )
        ]

    async def evaluate_async(self, evaluation_case):
        return self.evaluate(evaluation_case)


@pytest.fixture
def patch_run(monkeypatch):
    """Patch the runner's seams: sessions, judge model, evaluator resolution."""

    sessions = [
        _StubSession("s1", "Paris."),
        _StubSession("s2", "London."),
    ]
    monkeypatch.setattr("saes.run.runner.load_sessions", lambda cfg: sessions)
    monkeypatch.setattr("saes.run.runner.build_model", lambda cfg: object())
    monkeypatch.setattr(
        "saes.run.runner.resolve_evaluator",
        lambda ref, model: _FixedEvaluator(
            0.9 if "Helpful" in ref.id else 0.4, name=ref.id
        ),
    )
    return sessions


def _config():
    return EvaluationConfig(
        name="t",
        dataSource=DataSourceConfig(type="otlp_file", path="ignored.jsonl"),
        judge=JudgeModelConfig(
            provider="openai_compatible", model="m", base_url="https://x/v1"
        ),
        evaluators=[
            {"id": "Builtin.Helpfulness", "type": "builtin"},
            {"id": "Builtin.Correctness", "type": "builtin"},
        ],
    )


@pytest.mark.filterwarnings("ignore::UserWarning")  # stub Session isn't a native Session
async def test_run_on_demand_end_to_end(patch_run):
    result = await run_on_demand(_config())
    assert result.config_name == "t"
    assert result.judge_model == "m"
    assert set(result.session_ids) == {"s1", "s2"}
    assert result.evaluator_ids == ["Builtin.Helpfulness", "Builtin.Correctness"]
    # fixed scores flow through the native Experiment into aggregates
    assert result.aggregates["Builtin.Helpfulness"]["avg"] == pytest.approx(0.9)
    assert result.aggregates["Builtin.Correctness"]["avg"] == pytest.approx(0.4)
    assert result.aggregates["Builtin.Helpfulness"]["n"] == 2.0
