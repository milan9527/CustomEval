# Examples — verification runs, agent sources, and raw evidence

Real end-to-end runs of SAES against AWS (AgentCore Runtime → CloudWatch →
evaluation), preserved so the claims in the top-level docs are auditable. The
**analysis and conclusions** live in [../DOCUMENTATION.md](../DOCUMENTATION.md)
(§8 results, §10 verification); this directory holds the **reproducible scripts,
agent sources, and raw captured outputs** behind them.

> These are records of what was actually run. The scripts hardcode the original
> environment's paths (`/home/ec2-user/...`), a venv, and account-specific
> **runtime ids** (e.g. `saesstrands-ZhPiI77pEM`). To re-run against your own
> deployment, deploy the agents (below) and replace the runtime ids / paths with
> yours. No AWS account id or credentials are included. See
> [../WALKTHROUGH.md](../WALKTHROUGH.md) for the clean, generalized how-to.

## Agent sources (`agents/`)

The six agents used across the runs — the same `get_weather` + `calculate`
tool-calling scenario in five frameworks, plus a deliberately-bad one:

| Dir                         | Framework                 | Purpose                              |
| --------------------------- | ------------------------- | ------------------------------------ |
| `agents/strands_tools/`     | Strands                   | reference "good" agent (native OTEL) |
| `agents/noframework_tools/` | plain Python + boto3      | good, no framework                   |
| `agents/langgraph_tools/`   | LangGraph (OpenInference) | good                                 |
| `agents/crewai_tools/`      | CrewAI (OpenInference)    | good                                 |
| `agents/claudesdk_tools/`   | Claude Agent SDK          | good, manual OTEL (see below)        |
| `agents/bad_agent/`         | Strands                   | evasive system prompt → scores LOW   |

Each has `agent.py` + `requirements.txt` + `Dockerfile`. Deploy with the
AgentCore starter toolkit (see WALKTHROUGH.md §Part A); the deploy state
(`.bedrock_agentcore.yaml`, which carries your account id/role ARNs) is
intentionally **not** committed.

`claudesdk_tools/` is the **manual-contract** case (DOCUMENTATION.md §7.4 Path
B): the Claude Agent SDK calls Bedrock through a bundled CLI subprocess, so
AgentCore's botocore instrumentation captures nothing — the agent emits the
OTEL contract itself (root span with `session.id`/`gen_ai.prompt`/
`gen_ai.completion`, roled message events, Converse-shaped
`toolUse`/`toolResult` events per tool call). It also ships
`smoke_local.py`, which runs one invocation under an in-memory OTEL exporter
and produces a JSONL dump for `saes doctor` — the pre-deploy check the other
agents don't need. Verified on a real deployment (runtime `saesclaudesdk`,
2026-07-16): 3-turn session, 7/7 configured evaluators ran, tool-level 4/4
calls at 1.0, per-turn prompt/answer/tool pairing correct.

This agent was itself generated from the
[`otel-eval-contract` skill](../.claude/skills/otel-eval-contract/SKILL.md) —
the §7.4 contract packaged for Claude Code, so agents written with the repo
open follow the contract automatically instead of you restating it per agent.

## Claude Agent SDK — the OTLP→CloudWatch emit path (`agent_sdk/`)

The other Claude Agent SDK route (§7.4 Path A), and the one that proves the full
**OTLP → collector → CloudWatch → `saes eval`** path end to end: a **Claude
Agent SDK** (`claude-agent-sdk`) agent with its built-in OpenTelemetry,
exporting over OTLP to an ADOT collector that forwards to CloudWatch, then
evaluated on-demand. All 12 reference-free evaluators produced real scores over
3 live sessions. See **[agent_sdk/README.md](agent_sdk/README.md)** — it
includes the agent (`run_agent.py`), the collector config (`collector.yaml`),
the verbatim result (`AGENT_SDK_EVAL_OUTPUT.txt` / `.json`), and the honest
notes (the SDK exports no tool-result payload; the tool-parameter judge is
non-deterministic on this case). This is the positive counterpart to the
raw-`anthropic`-SDK negative result (bypasses botocore, emits nothing).

## Scripts (each prints real judge scores)

| Script                               | What it demonstrates                                                 | Raw output                                                                              |
| ------------------------------------ | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `builtin_suite.py`                   | every built-in discriminates good>bad (14/14)                        | `BUILTIN_SUITE_OUTPUT.txt`                                                              |
| `bad_examples.py`                    | deliberately-bad multi-turn sessions score 0.0                       | `BAD_EXAMPLES_OUTPUT.txt`                                                               |
| `framework_matrix.py`                | 4 frameworks × 15 evaluators over real CloudWatch traces (all 15/15) | `FRAMEWORK_MATRIX_OUTPUT.txt`                                                           |
| `framework_matrix_verbose.py`        | same, step-by-step with span types + judge reasoning                 | `FRAMEWORK_MATRIX_TRANSCRIPT.txt`                                                       |
| `goodbad_run.sh` + `goodbad_eval.py` | good vs. bad **multi-turn** agents across frameworks                 | `GOODBAD_TRANSCRIPT.txt`, `GOODBAD_EVAL_OUTPUT.txt`, `GOODBAD_EVAL_MULTITURN_FIXED.txt` |
| `my_agent.py`                        | no-framework agent → local OTEL dump (offline)                       | —                                                                                       |

## Realistic customer scenarios (`complex_agents/`)

The most customer-representative example: four **realistic** agents (SaaS
helpdesk, HR RAG assistant, travel-booking workflow, clinic compliance desk),
each deployed to AgentCore, driven with real multi-turn conversations, and
evaluated on-demand with per-scenario evaluators + ground truth + a **custom
code evaluator**. See **[complex_agents/COMPLEX_SCENARIOS.md](complex_agents/COMPLEX_SCENARIOS.md)**
— it surfaces a genuine quality issue (an over-cautious support agent that fails
to complete an authorized refund) alongside faithful-RAG, full booking-workflow,
and compliance-refusal results.

## Records (verbatim commands + outputs)

- **GOODBAD_MULTITURN.md** — good vs. bad multi-turn evaluation across
  frameworks: the discrimination result, the judge's reasoning, and the honest
  findings (incl. the multi-turn turn-pairing fix). **Start here** for the
  most recent, most complete example.
- **RUN_LOG.md** — Part 1 (offline: agent → local dump → `saes run`) and Part 2
  (online: AgentCore agent → CloudWatch → `saes serve` → results to CloudWatch).
- **RUN_LOG_PART3.md** — multi-framework agents on AgentCore + the tool-trajectory
  supplement; §B–F are a live verbatim re-run.
- **`*_TRANSCRIPT.txt` / `*_OUTPUT.txt`** — raw captured terminal output backing
  the records.

## Environment note

The runs used an editable SAES install and a Bedrock OpenAI-compatible judge:

```bash
pip install -e '.[dev]' openai aws-bedrock-token-generator
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
```
