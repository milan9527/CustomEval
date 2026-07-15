# Complex customer agents — realistic multi-scenario on-demand evaluation

Four realistic customer agents, each deployed to AgentCore Runtime, driven with
real multi-turn conversations, and evaluated **on-demand** with SAES over their
real CloudWatch traces (real Bedrock judge) — using per-scenario built-in
evaluators + ground truth + a **custom code evaluator**. This is the closest
example to how a customer evaluates their own production agents.

**Full write-up (design, transcripts, scores, judge reasoning, honest findings):
[COMPLEX_SCENARIOS.md](COMPLEX_SCENARIOS.md).**

## The four agents (`agents/`)

| Dir | Domain | Tools | Exercises |
|---|---|---|---|
| `support_agent` | SaaS helpdesk | account/invoice lookup, refund, escalate | GoalSuccessRate, InstructionFollowing, tool use |
| `rag_agent` | HR policy assistant | knowledge-base search | Faithfulness, ContextRelevance, Correctness (no-hallucination) |
| `booking_agent` | travel workflow | flight/hotel search + book | ToolSelection/ToolParameter, TrajectoryMatch, GoalSuccessRate |
| `compliance_agent` | clinic front desk | appointment lookup/reschedule | Refusal, Harmfulness, Stereotyping + a **custom** compliance code evaluator |

Each has `agent.py` + `requirements.txt` + `Dockerfile` (Strands, native OTEL →
CloudWatch). Deploy state (`.bedrock_agentcore.yaml`, account id / role ARNs) is
intentionally **not** committed — see [../../WALKTHROUGH.md](../../WALKTHROUGH.md)
for the deploy procedure.

## Reproduce

```bash
# 1. deploy each agent (from its dir): agentcore configure + deploy  (see WALKTHROUGH.md)
# 2. drive the conversations:
bash run_scenarios.sh              # → COMPLEX_TRANSCRIPT.txt
# 3. evaluate on-demand (set the runtime ids in complex_eval.py to your deployed ones):
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
python complex_eval.py             # → COMPLEX_EVAL_OUTPUT.txt
```

> The runtime ids in `complex_eval.py` / `run_scenarios.sh` are the ones from the
> recorded run — replace with your own deployed ids.

## Headline finding

The evaluators track **real quality**, not just "did it answer". The support
agent scored **InstructionFollowing 0.0 / Helpfulness 0.167** on the turn where
the user said *"go ahead with the refund"* but the agent asked yet another
clarifying question instead of acting — a genuine, actionable issue (over-cautious
agent fails to complete an authorized action), flagged automatically. Meanwhile
RAG stayed faithful on an out-of-KB question (no hallucination), booking hit 1.0
end-to-end incl. trajectory match, and the compliance agent correctly refused
medical-dosage advice (built-in Harmfulness + a custom code evaluator both 1.0).
See COMPLEX_SCENARIOS.md for the full table and the honest last-turn/polarity
caveats.
