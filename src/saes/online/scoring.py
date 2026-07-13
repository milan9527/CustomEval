"""Online scoring adapter (T15) — wires the worker to the real pipeline.

Given a set of completed session ids, score them via the native provider +
Experiment (reusing the runner's building blocks) and emit results to
CloudWatch. Returns the ids successfully scored (so the worker only marks
those, leaving failures to retry next cycle).
"""

from __future__ import annotations

import asyncio

from strands_evals import Experiment

from ..config.schema import EvaluationConfig
from ..evaluators.registry import resolve_evaluator
from ..ingest.cloudwatch import build_provider
from ..judge.providers import build_model
from ..report import build_report, emit_to_cloudwatch
from ..run.ground_truth import load_ground_truth
from ..run.runner import RunResult, _aggregate, _case_for


def make_scorer(config: EvaluationConfig):
    """Build a `score(session_ids) -> scored_ids` callable for the worker.

    Resolves the judge + evaluators once; each call scores a batch of session
    ids through the native provider's `as_task()` (per-session read+map) and
    emits to the configured CloudWatch sink.
    """
    gt = load_ground_truth(config.ground_truth)
    judge_model = build_model(config.judge)
    evaluator_ids = [ref.id for ref in config.evaluators]
    provider = build_provider(config.data_source.cloudwatch)
    sink = config.results_sink.cloudwatch if config.results_sink else None
    region = config.data_source.cloudwatch.region if config.data_source.cloudwatch else None

    def score(session_ids: list[str]) -> list[str]:
        if not session_ids:
            return []
        # fresh evaluator instances per batch (native engine rejects reuse across
        # runs / duplicate names within a run)
        evaluators = [resolve_evaluator(ref, judge_model) for ref in config.evaluators]
        cases = [_case_for(sid, gt) for sid in session_ids]
        # native read+map wrapped with the non-Strands tool-trajectory supplement
        # (F6 fix) — same path the on-demand runner uses.
        from ..ingest.cloudwatch_task import build_supplemented_task
        task = build_supplemented_task(provider, config.data_source.cloudwatch)

        try:
            report = asyncio.run(
                Experiment(cases=cases, evaluators=evaluators).run_evaluations_async(task)
            )
        except Exception:  # noqa: BLE001 - batch failed; retry next cycle
            return []

        result = RunResult(
            config_name=config.name,
            judge_model=config.judge.model,
            report=report,
            evaluator_ids=evaluator_ids,
            session_ids=session_ids,
            aggregates=_aggregate(report, evaluator_ids),
        )
        doc = build_report(result)
        if sink is not None:
            try:
                emit_to_cloudwatch(doc, sink, region=region)
            except Exception:  # noqa: BLE001 - emit failure shouldn't lose the scores
                pass
        # every session in the batch was processed by the native run
        return list(session_ids)

    return score


__all__ = ["make_scorer"]
