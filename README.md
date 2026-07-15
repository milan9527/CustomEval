# SAES — Strands Agent Evaluation Suite

> 中文版见 [README.zh.md](README.zh.md)。

Open-source evaluation for AI agents, built with the [Strands Agents SDK](https://strandsagents.com/) and integrated with Amazon Bedrock AgentCore Observability.

- **Bring your own judge** — any OpenAI-compatible endpoint with tool-calling / structured-output support (OpenAI, Azure, vLLM, LiteLLM, Bedrock) serves as the LLM-as-a-Judge. `saes doctor --judge` verifies it before you run.
- **Framework-agnostic** — evaluates any agent that emits OpenTelemetry GenAI traces to CloudWatch, regardless of SDK or language. Strands, LangGraph, CrewAI, and no-framework agents all reach **all 15 built-in evaluators** — the adaptation lives in SAES ingestion, not in your agent.
- **AgentCore-parity evaluators** — 13 built-ins + deterministic trajectory scorers + custom LLM/code evaluators, all native `strands-agents-evals`, so scores line up with managed AgentCore Evaluations.

## Quick start

Already have an agent on **AgentCore Runtime**? Evaluate it with one command —
just the runtime id. No YAML, no ground truth, no trace plumbing. Requires
**Python 3.12**:

```bash
git clone https://github.com/milan9527/CustomEval.git && cd CustomEval
python3.12 -m venv .venv && source .venv/bin/activate    # activate FIRST
pip install -e '.[dev]' openai aws-bedrock-token-generator

# judge = Amazon Bedrock, via your AWS credentials — no external key
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"

saes eval myagent-XXXXXXXXXX --html out/report.html      # ← your AgentCore Runtime id
#   scans the last 7 days by default; add --days 30 for older sessions
#   evaluating /aws/bedrock-agentcore/runtimes/myagent-XXXXXXXXXX-DEFAULT
#     Builtin.Helpfulness        avg=0.833  pass=100%  n=1
#     Builtin.Coherence          avg=1.000  pass=100%  n=1
#     ...
```

`saes eval` derives the runtime's CloudWatch log group, discovers its sessions,
and scores them with the 12 reference-free built-in evaluators. Options mirror
AgentCore Evaluations:

```bash
saes eval --list-evaluators                            # all 13 built-ins + trajectory matchers
saes eval myagent-XXX -e Builtin.Helpfulness,Builtin.Harmfulness   # pick evaluators
saes eval myagent-XXX --all                            # every built-in
saes eval myagent-XXX --sampling 25                    # score 25% of sessions
```

**Continuous (online) monitoring** is the same one-liner — just `serve` instead
of `eval`:

```bash
saes serve myagent-XXX                                 # poll live traffic, score completed sessions
saes serve myagent-XXX --once                          # a single cycle (CI/cron)
```

Full start-to-finish example (build agent → deploy → evaluate) in
**[WALKTHROUGH.md](WALKTHROUGH.md)**.

No agent yet? Score a bundled trace sample in 1 minute — see
[DOCUMENTATION.md §4.0](DOCUMENTATION.md#40-i-just-cloned-this-repo-and-i-have-my-own-agent--where-do-i-start).

## Documentation

- **[WALKTHROUGH.md](WALKTHROUGH.md)** — the complete linear example: clone →
  build agent → deploy to AgentCore → CloudWatch → evaluate. **Start here.**
- **[DOCUMENTATION.md](DOCUMENTATION.md)** — the full reference in one place:
  project description, architecture, all usage paths, configuration, the
  evaluator catalog, per-framework support, evaluation scenarios + results
  analysis, online evaluation, and the verification log.
- **[SPEC.md](SPEC.md)** — the full technical specification.
- **[examples/](examples/)** — deployable agent sources and the **raw records**
  of the real runs behind the results: the framework matrix, good-vs-bad
  multi-turn, discrimination suites, and
  **[realistic customer scenarios](examples/complex_agents/)** (SaaS helpdesk /
  RAG / booking / compliance, evaluated on-demand with ground truth + a custom
  evaluator).

## Status

M1–M3 complete (offline evaluation, CloudWatch ingestion, online worker, CDK
dashboard). 198 unit tests. Verified end-to-end with real Bedrock judges (offline
+ online), a real deployed AgentCore Runtime agent, and four frameworks (Strands,
LangGraph, CrewAI, no-framework) — all reaching 15/15 evaluators. Not yet released.

## License

Apache-2.0
