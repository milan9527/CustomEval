"""Deliberately-bad multi-turn agent sessions, evaluated with a REAL Bedrock
judge, to prove SAES evaluators discriminate (low scores for bad behavior).

Each scenario targets a specific evaluator's failure mode. Run:
    export SAES_JUDGE_API_KEY=$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')
    python bad_examples.py
"""
import asyncio, os
from datetime import datetime, timezone

from strands_evals import Case, Experiment
from strands_evals.types.trace import AgentInvocationSpan, Session, SpanInfo, Trace

from saes.config.schema import EvaluatorRef, JudgeModelConfig
from saes.evaluators import resolve_evaluator
from saes.judge.providers import build_model

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_session(sid, turns):
    """turns: [(user, assistant), ...] → a multi-turn native Session."""
    spans = []
    for i, (u, a) in enumerate(turns):
        si = SpanInfo(trace_id="t0", span_id=f"s{i}", session_id=sid, start_time=_NOW, end_time=_NOW)
        spans.append(AgentInvocationSpan(span_info=si, user_prompt=u, agent_response=a,
                                         available_tools=[], system_prompt=""))
    return Session(session_id=sid, traces=[Trace(spans=spans, trace_id="t0", session_id=sid)])


# (id, evaluator, ground_truth, turns, why-it-should-score-low)
SCENARIOS = [
    ("wrong-fact", "Builtin.Correctness",
     {"expected_output": "Paris"},
     [("What is the capital of France?", "The capital of France is Berlin.")],
     "factually wrong answer"),

    ("unhelpful", "Builtin.Helpfulness", {},
     [("How do I reset my password?", "Passwords are important for security."),
      ("But HOW do I reset it?", "You should use a strong password."),
      ("You're not answering me.", "Security is a shared responsibility.")],
     "3 turns, never answers the actual question"),

    ("self-contradiction", "Builtin.Coherence", {},
     [("Is the store open on Sunday?", "Yes, we're open every day of the week."),
      ("Great, what are Sunday hours?", "We're closed on Sundays.")],
     "contradicts its own earlier answer across turns"),

    ("unfaithful", "Builtin.Faithfulness", {},
     [("My order id is 5, look it up.", "Order 5 is a red shirt, shipped Monday."),
      ("Remind me what I ordered?", "You ordered a blue jacket, delivered Friday.")],
     "later turn conflicts with facts it stated earlier"),

    ("ignores-instructions", "Builtin.InstructionFollowing", {},
     [("Answer in exactly one word: what color is the sky?",
       "The sky appears blue due to Rayleigh scattering of sunlight in the atmosphere, "
       "which scatters shorter blue wavelengths more than longer red ones.")],
     "explicit one-word instruction ignored with a long answer"),

    ("goal-not-met", "Builtin.GoalSuccessRate",
     {"expected_assertion": "The agent booked a flight to Tokyo and confirmed the date."},
     [("Book me a flight to Tokyo next Friday.", "Tokyo is a wonderful city to visit!"),
      ("So is it booked?", "You'll love the cherry blossoms in spring.")],
     "user goal (book a flight) never accomplished"),

    # --- GOOD counterparts (should score HIGH) — proves the judge discriminates,
    #     not just "everything scores low" ---
    ("good-correct", "Builtin.Correctness",
     {"expected_output": "Paris"},
     [("What is the capital of France?", "The capital of France is Paris.")],
     "correct answer — expect HIGH"),

    ("good-goal-met", "Builtin.GoalSuccessRate",
     {"expected_assertion": "The agent booked a flight to Tokyo and confirmed the date."},
     [("Book me a flight to Tokyo next Friday.",
       "Done — I've booked your flight to Tokyo departing next Friday, Jan 9. "
       "Confirmation code TYO-4471. Anything else?")],
     "goal accomplished — expect HIGH"),
]


async def main():
    judge = build_model(JudgeModelConfig(provider="bedrock", model=os.environ["BEDROCK_MODEL_ID"]))
    print(f"{'scenario':22s} {'evaluator':32s} {'score':>6s}  label")
    print("-" * 78)
    for sid, evid, gt, turns, why in SCENARIOS:
        session = make_session(sid, turns)
        ev = resolve_evaluator(EvaluatorRef(id=evid, type="builtin"), judge)
        case = Case(
            name=sid, input=sid, session_id=sid,
            expected_output=gt.get("expected_output"),
            expected_assertion=gt.get("expected_assertion"),
        )
        final = session.traces[0].spans[-1].agent_response
        rep = await Experiment(cases=[case], evaluators=[ev]).run_evaluations_async(
            lambda c, s=session, f=final: {"output": f, "trajectory": s})
        o = rep.detailed_results[0][0]
        expect_high = sid.startswith("good-")
        ok = (o.score > 0.5) if expect_high else (o.score <= 0.5)
        flag = ("✓ high" if expect_high else "✓ low") if ok else "✗ UNEXPECTED"
        print(f"{sid:22s} {evid:32s} {o.score:6.3f}  {o.label}   [{flag}]")
        print(f"   why: {why}")
        print(f"   judge: {o.reason[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
