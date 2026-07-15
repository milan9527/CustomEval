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

The five agents used across the runs — the same `get_weather` + `calculate`
tool-calling scenario in four frameworks, plus a deliberately-bad one:

| Dir | Framework | Purpose |
|---|---|---|
| `agents/strands_tools/` | Strands | reference "good" agent (native OTEL) |
| `agents/noframework_tools/` | plain Python + boto3 | good, no framework |
| `agents/langgraph_tools/` | LangGraph (OpenInference) | good |
| `agents/crewai_tools/` | CrewAI (OpenInference) | good |
| `agents/bad_agent/` | Strands | evasive system prompt → scores LOW |

Each has `agent.py` + `requirements.txt` + `Dockerfile`. Deploy with the
AgentCore starter toolkit (see WALKTHROUGH.md §Part A); the deploy state
(`.bedrock_agentcore.yaml`, which carries your account id/role ARNs) is
intentionally **not** committed.

## Scripts (each prints real judge scores)

| Script | What it demonstrates | Raw output |
|---|---|---|
| `builtin_suite.py` | every built-in discriminates good>bad (14/14) | `BUILTIN_SUITE_OUTPUT.txt` |
| `bad_examples.py` | deliberately-bad multi-turn sessions score 0.0 | `BAD_EXAMPLES_OUTPUT.txt` |
| `framework_matrix.py` | 4 frameworks × 15 evaluators over real CloudWatch traces (all 15/15) | `FRAMEWORK_MATRIX_OUTPUT.txt` |
| `framework_matrix_verbose.py` | same, step-by-step with span types + judge reasoning | `FRAMEWORK_MATRIX_TRANSCRIPT.txt` |
| `goodbad_run.sh` + `goodbad_eval.py` | good vs. bad **multi-turn** agents across frameworks | `GOODBAD_TRANSCRIPT.txt`, `GOODBAD_EVAL_OUTPUT.txt`, `GOODBAD_EVAL_MULTITURN_FIXED.txt` |
| `my_agent.py` | no-framework agent → local OTEL dump (offline) | — |

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
