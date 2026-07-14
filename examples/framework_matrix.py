"""4 frameworks × 14 built-in evaluators matrix, over their REAL AgentCore
CloudWatch traces (weather+math tool-calling sessions). Real Bedrock judge.

Uses the SAES supplemented CloudWatch task so tool trajectories work for the
non-Strands agents. Records a score per (framework, evaluator).

    export SAES_JUDGE_API_KEY=$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')
    python framework_matrix.py
"""
import asyncio, os

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

# The 14 built-ins. Ground truth appropriate to the weather+math tool sessions.
# expected_trajectory for tool/trajectory evaluators; expected_output/assertion
# where the evaluator needs it. Others run reference-free.
EVALUATORS = [
    ("Builtin.Helpfulness", {}),
    ("Builtin.Correctness", {"expected_output": "The weather in Tokyo is 22C; 15% of 240 is 36; 12x8 is 96."}),
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


async def eval_agent(judge, fw, log_suffix):
    cfg = CloudWatchSource(log_group_names=[f"/aws/bedrock-agentcore/runtimes/{log_suffix}"],
                           region="us-east-1", lookback_days=3)
    provider = build_provider(cfg)
    sids = discover_session_ids(provider, cfg)
    if not sids:
        return {e: None for e, _ in EVALUATORS}
    sid = sids[0]
    task = build_supplemented_task(provider, cfg)
    scores = {}
    for evid, gt in EVALUATORS:
        ev = resolve_evaluator(EvaluatorRef(id=evid, type="builtin"), judge)
        case = Case(name=sid, input=sid, session_id=sid,
                    expected_output=gt.get("expected_output"),
                    expected_assertion=gt.get("expected_assertion"),
                    expected_trajectory=gt.get("expected_trajectory"))
        try:
            rep = await Experiment(cases=[case], evaluators=[ev]).run_evaluations_async(task)
            outs = rep.detailed_results[0]
            scores[evid] = outs[0].score if outs else None
        except Exception as e:  # noqa: BLE001
            scores[evid] = f"ERR:{type(e).__name__}"
    return scores


async def main():
    judge = build_model(JudgeModelConfig(provider="bedrock", model=os.environ["BEDROCK_MODEL_ID"]))
    results = {}
    for fw, suffix in AGENTS.items():
        print(f"# evaluating {fw} ...", flush=True)
        results[fw] = await eval_agent(judge, fw, suffix)

    # print matrix
    evs = [e for e, _ in EVALUATORS]
    header = "evaluator".ljust(32) + "".join(f"{fw:>10s}" for fw in AGENTS)
    print("\n" + header)
    print("-" * len(header))
    for evid in evs:
        row = evid.ljust(32)
        for fw in AGENTS:
            v = results[fw].get(evid)
            row += (f"{v:>10.3f}" if isinstance(v, float) else f"{str(v):>10s}")
        print(row)


if __name__ == "__main__":
    asyncio.run(main())
