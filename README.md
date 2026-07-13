# SAES — Strands Agent Evaluation Suite

Open-source evaluation for AI agents, built with the [Strands Agents SDK](https://strandsagents.com/) and integrated with Amazon Bedrock AgentCore Observability.

- **Bring your own judge** — any OpenAI-compatible endpoint with tool-calling / structured-output support (OpenAI, Azure, vLLM, LiteLLM, Bedrock) serves as the LLM-as-a-Judge. `saes doctor --judge` verifies it before you run.
- **Framework-agnostic** — evaluates any agent that emits OpenTelemetry GenAI traces to CloudWatch, regardless of SDK or language. Strands, LangGraph, CrewAI, and no-framework agents all reach **all 15 built-in evaluators** — the adaptation lives in SAES ingestion, not in your agent.
- **AgentCore-parity evaluators** — 13 built-ins + deterministic trajectory scorers + custom LLM/code evaluators, all native `strands-agents-evals`, so scores line up with managed AgentCore Evaluations.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # activate FIRST
pip install -e '.[dev]' openai
saes init --agent-type rag --out eval.yaml   # scaffold a config
saes doctor --data-source traces.jsonl       # check your OTEL traces
saes doctor --judge eval.yaml                # verify your judge endpoint
saes run -c eval.yaml --html out/report.html # evaluate (exit non-zero if a gate fails)
```

## Documentation

**Everything is in one place: [DOCUMENTATION.md](DOCUMENTATION.md)** — project
description, architecture, end-to-end usage (from building an agent to a scored
report), configuration, the evaluator catalog, per-framework support, evaluation
scenarios + results analysis, online evaluation, and the verification log.

The full technical specification remains in [SPEC.md](SPEC.md).

## Status

M1–M3 complete (offline evaluation, CloudWatch ingestion, online worker, CDK
dashboard). 186 unit tests. Verified end-to-end with real Bedrock judges (offline
+ online), a real deployed AgentCore Runtime agent, and four frameworks (Strands,
LangGraph, CrewAI, no-framework) — all reaching 15/15 evaluators. Not yet released.

## License

Apache-2.0
