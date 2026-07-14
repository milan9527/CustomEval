"""Evaluate the good-vs-bad multi-turn sessions produced by goodbad_run.sh.

Scores each specific session (targeted by id) with reference-free built-ins +
tool-level evaluators, via the SAES supplemented CloudWatch task and a real
Bedrock OpenAI-compatible judge. Records per-evaluator score + judge reasoning
so the good/bad contrast is auditable.

    export SAES_JUDGE_API_KEY=$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')
    python goodbad_eval.py
"""
import asyncio
import os

from strands_evals import Case, Experiment

from saes.config.schema import CloudWatchSource, EvaluatorRef, JudgeModelConfig
from saes.evaluators import resolve_evaluator
from saes.ingest.cloudwatch import build_provider
from saes.ingest.cloudwatch_task import build_supplemented_task
from saes.judge.providers import build_model

# (label, runtime log-group suffix, session id) — from GOODBAD_TRANSCRIPT.txt
TARGETS = [
    ("good-strands",   "saesstrands-ZhPiI77pEM-DEFAULT",   "goodstrands1784015164sessionpadding123456"),
    ("good-noframe",   "saesnoframe-6AXcAT2oW4-DEFAULT",   "goodnoframe1784015164sessionpadding123456"),
    ("good-langgraph", "saeslanggraph-vSzHF7G235-DEFAULT", "goodlanggraph1784015164sessionpadding123456"),
    ("good-crewai",    "saescrewai-JjA6Jp5dHw-DEFAULT",    "goodcrewai1784015164sessionpadding123456"),
    ("bad-strands",    "saesbad-xODWpQ4r1o-DEFAULT",       "badstrands1784015164sessionpadding123456"),
]

# reference-free built-ins that most clearly separate good vs bad on this scenario
EVALUATORS = [
    "Builtin.Helpfulness",
    "Builtin.InstructionFollowing",
    "Builtin.ResponseRelevance",
    "Builtin.Coherence",
    "Builtin.GoalSuccessRate",
]
GOAL = "The agent answered the weather and math questions using its tools."


async def eval_session(judge, label, suffix, sid):
    cfg = CloudWatchSource(log_group_names=[f"/aws/bedrock-agentcore/runtimes/{suffix}"],
                           region="us-east-1", lookback_days=1)
    provider = build_provider(cfg)
    task = build_supplemented_task(provider, cfg)
    rows = []
    for evid in EVALUATORS:
        ev = resolve_evaluator(EvaluatorRef(id=evid, type="builtin"), judge)
        case = Case(name=sid, input=sid, session_id=sid, expected_assertion=GOAL)
        try:
            rep = await Experiment(cases=[case], evaluators=[ev]).run_evaluations_async(task)
            outs = rep.detailed_results[0]
            o = outs[0] if outs else None
            if o is None:
                rows.append((evid, None, "(no result)"))
            else:
                rows.append((evid, o.score, (o.reason or "").replace("\n", " ")[:150]))
        except Exception as e:  # noqa: BLE001
            rows.append((evid, f"ERR:{type(e).__name__}", str(e)[:80]))
    return rows


async def main():
    judge = build_model(JudgeModelConfig(
        provider="openai_compatible", model="openai.gpt-oss-20b-1:0",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
        api_key_env="SAES_JUDGE_API_KEY"))
    # quiet the judge tool-retry noise
    import logging
    logging.getLogger("strands.tools.executors._executor").setLevel(logging.CRITICAL)

    all_rows = {}
    for label, suffix, sid in TARGETS:
        print(f"\n########## {label}  (session {sid[:24]}...) ##########", flush=True)
        rows = await eval_session(judge, label, suffix, sid)
        all_rows[label] = rows
        for evid, score, reason in rows:
            s = f"{score:.3f}" if isinstance(score, float) else str(score)
            print(f"  {evid:30s} {s:>8s}   {reason}", flush=True)

    # compact good-vs-bad summary
    print("\n\n=== SUMMARY: score by evaluator (good agents vs bad-strands) ===")
    hdr = "evaluator".ljust(30) + "".join(f"{l.replace('good-','').replace('bad-','!'):>11s}" for l, _, _ in TARGETS)
    print(hdr); print("-" * len(hdr))
    for i, evid in enumerate(EVALUATORS):
        line = evid.ljust(30)
        for label, _, _ in TARGETS:
            v = all_rows[label][i][1]
            line += (f"{v:>11.3f}" if isinstance(v, float) else f"{str(v):>11s}")
        print(line)
    print("\n(! = bad-strands. Lower scores on the bad agent = evaluators discriminate.)")


if __name__ == "__main__":
    asyncio.run(main())
