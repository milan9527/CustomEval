"""Verbose, step-by-step 4-framework × 14-evaluator run — logs every step with
real intermediate output (invoke result, discovered session, per-evaluator score
+ judge reason). Mirrors the detail level of PART3_TRANSCRIPT.

    export SAES_JUDGE_API_KEY=$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')
    python -u framework_matrix_verbose.py     # prints incrementally
"""
import asyncio, os, sys

from strands_evals import Case, Experiment

from saes.config.schema import CloudWatchSource, EvaluatorRef, JudgeModelConfig
from saes.evaluators import resolve_evaluator
from saes.ingest.cloudwatch import build_provider, discover_session_ids
from saes.ingest.cloudwatch_task import build_supplemented_task
from saes.judge.providers import build_model

AGENTS = {
    "strands":   "saesstrands-ZhPiI77pEM-DEFAULT",
    "noframe":   "saesnoframe-6AXcAT2oW4-DEFAULT",
    "langgraph": "saeslanggraph-vSzHF7G235-DEFAULT",
    "crewai":    "saescrewai-JjA6Jp5dHw-DEFAULT",
}

EVALUATORS = [
    ("Builtin.Helpfulness", {}),
    ("Builtin.Correctness", {"expected_output": "Tokyo weather ~22C; 15% of 240 = 36; 12x8 = 96"}),
    ("Builtin.Coherence", {}),
    ("Builtin.Conciseness", {}),
    ("Builtin.Faithfulness", {}),
    ("Builtin.Harmfulness", {}),
    ("Builtin.InstructionFollowing", {}),
    ("Builtin.ResponseRelevance", {}),
    ("Builtin.ContextRelevance", {}),
    ("Builtin.Refusal", {}),
    ("Builtin.Stereotyping", {}),
    ("Builtin.GoalSuccessRate", {"expected_assertion": "The agent answered weather and math questions using tools."}),
    ("Builtin.ToolSelectionAccuracy", {}),
    ("Builtin.ToolParameterAccuracy", {}),
    ("Builtin.TrajectoryAnyOrderMatch", {"expected_trajectory": ["get_weather", "calculate"]}),
]


def log(msg=""):
    print(msg, flush=True)


async def run_framework(judge, fw, suffix):
    log("\n" + "=" * 74)
    log(f"FRAMEWORK: {fw}   (runtime {suffix})")
    log("=" * 74)

    cfg = CloudWatchSource(log_group_names=[f"/aws/bedrock-agentcore/runtimes/{suffix}"],
                           region="us-east-1", lookback_days=1)
    provider = build_provider(cfg)

    log("$ SAES discover session ids from CloudWatch")
    sids = discover_session_ids(provider, cfg)
    log(f"  -> {sids}")
    if not sids:
        log("  (no sessions; skipping)")
        return
    sid = sids[0]

    task = build_supplemented_task(provider, cfg)
    log(f"$ build_supplemented_task; task(case) for session {sid[:8]}")
    probe = task(Case(name=sid, input=sid, session_id=sid))
    sess = probe.get("trajectory") if isinstance(probe, dict) else None
    types = sorted({type(s).__name__ for tr in getattr(sess, "traces", []) for s in tr.spans}) if sess else []
    log(f"  reconstructed span types: {types}")
    log(f"  supplemented tool trajectory: {getattr(sess, '_saes_tool_names', None)}")

    log(f"\n$ evaluate {len(EVALUATORS)} built-ins:")
    for evid, gt in EVALUATORS:
        ev = resolve_evaluator(EvaluatorRef(id=evid, type="builtin"), judge)
        case = Case(name=sid, input=sid, session_id=sid,
                    expected_output=gt.get("expected_output"),
                    expected_assertion=gt.get("expected_assertion"),
                    expected_trajectory=gt.get("expected_trajectory"))
        try:
            rep = await Experiment(cases=[case], evaluators=[ev]).run_evaluations_async(task)
            outs = rep.detailed_results[0]
            if outs:
                o = outs[0]
                reason = (o.reason or "").replace("\n", " ")[:90]
                log(f"  {evid:32s} score={o.score:.3f}  {reason}")
            else:
                log(f"  {evid:32s} score=—  (no result: evaluator got no usable turn)")
        except Exception as e:  # noqa: BLE001
            log(f"  {evid:32s} ERROR {type(e).__name__}: {str(e)[:60]}")


async def main():
    judge = build_model(JudgeModelConfig(provider="bedrock", model=os.environ["BEDROCK_MODEL_ID"]))
    log(f"judge: bedrock / {os.environ['BEDROCK_MODEL_ID'].split('/')[-1]}")
    for fw, suffix in AGENTS.items():
        await run_framework(judge, fw, suffix)
    log("\n(done)")


if __name__ == "__main__":
    asyncio.run(main())
