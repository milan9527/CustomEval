# SAES — Strands Agent Evaluation Suite

> 中文版见 [DOCUMENTATION.zh.md](DOCUMENTATION.zh.md)。

**Open-source evaluation for AI agents, built with the [Strands Agents SDK](https://strandsagents.com/) and integrated with Amazon Bedrock AgentCore Observability.**

This is the single, consolidated reference for SAES. It covers what the project
is, how it's built, how to use it end-to-end (from building an agent to a scored
report), the frameworks and evaluation scenarios it was verified against, and an
analysis of the real results.

> Supersedes the earlier scattered docs (README, ARCHITECTURE, USAGE,
> VERIFICATION, FRAMEWORK_MATRIX, BUILTIN_SUITE, BAD_EXAMPLES, RUN_LOG,
> REPRODUCE, MULTIFRAMEWORK_RESULTS). The full technical spec remains in
> [SPEC.md](SPEC.md).

---

## Table of contents

1. [What SAES is](#1-what-saes-is)
2. [The two differentiators](#2-the-two-differentiators)
3. [Architecture & implementation](#3-architecture--implementation)
4. [End-to-end usage: from building an agent to a scored report](#4-end-to-end-usage-from-building-an-agent-to-a-scored-report)
   — start at [§4.0 "I just cloned this repo and I have my own agent"](#40-i-just-cloned-this-repo-and-i-have-my-own-agent--where-do-i-start)
5. [Configuration reference](#5-configuration-reference)
6. [The evaluator catalog](#6-the-evaluator-catalog)
7. [Framework support: how any framework reaches full coverage](#7-framework-support-how-any-framework-reaches-full-coverage)
   — incl. [§7.4 what YOUR agent must emit (the OTEL contract, by framework)](#74-what-your-agent-must-emit--the-otel-contract-by-framework)
8. [Evaluation scenarios & results analysis](#8-evaluation-scenarios--results-analysis)
   — incl. [§8.5 the evaluation, step by step](#85-the-evaluation-step-by-step)
9. [Online / production evaluation](#9-online--production-evaluation)
10. [Verification log: what was proven, and the bugs found](#10-verification-log-what-was-proven-and-the-bugs-found)
11. [Reproduce](#11-reproduce)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What SAES is

SAES is a self-hostable evaluation solution for AI agents. It reads the
OpenTelemetry (OTEL) traces your agent already emits, reconstructs each
conversation, and scores it with a catalog of evaluators — LLM-as-a-Judge and
deterministic — that mirror [Amazon Bedrock AgentCore Evaluations](https://aws.amazon.com/blogs/machine-learning/build-reliable-ai-agents-with-amazon-bedrock-agentcore-evaluations/).
It runs offline (a local trace dump, for CI/regression) or online (sampling a
live agent's CloudWatch traffic), and writes results back to CloudWatch as
metrics + structured logs so quality appears alongside operational signals.

**Design stance — reuse over rebuild.** The evaluation _engine_ is native
[`strands-agents-evals`](https://strandsagents.com/): the built-in evaluators,
trajectory scorers, prompt templates, session mappers, trace providers, and the
`Experiment`/`Case`/`Report` orchestration. SAES adds only the thin layers that
package lacks:

1. **OpenAI-compatible judge selection** as a first-class config surface.
2. A **framework-agnostic CloudWatch/OTEL ingestion adapter** (the part that
   makes _any_ agent evaluable, not just Strands).
3. An **online sampling worker** + CloudWatch results emission.

SAES never imports or runs your agent's code. The only integration contract is
the trace format.

### What it is not

- Not a replacement for AgentCore Observability's telemetry pipeline — it
  _consumes_ that pipeline.
- Not a managed service — it's a library + CLI + optional worker.
- Not an agent-authoring framework — it evaluates agents, it doesn't build them.
- Not limited to Strands agents — the name reflects the SDK it is _built with_,
  not what it can _evaluate_.

### Status

M1 (core offline evaluation), M2 (CloudWatch ingestion, EMF/JSON results, full
evaluator catalog), and M3 (online worker, Lambda code evaluators, dashboard/
alarms CDK) are complete. **199 unit tests passing** (+ CDK synth tests). Verified
end-to-end with real Bedrock judges (offline and online), against a real deployed
AgentCore Runtime agent, and across four frameworks (Strands, LangGraph, CrewAI,
no-framework). Apache-2.0. Not yet released.

---

## 2. The two differentiators

### 2.1 Bring your own judge

The LLM-as-a-Judge is **any OpenAI-compatible endpoint that supports tool calling
/ structured output**, or Amazon Bedrock. That means OpenAI, Azure OpenAI,
self-hosted vLLM (guided decoding), LiteLLM, SageMaker, or Bedrock's
OpenAI-compatible API — you are not locked into a managed judge.

> **Hard requirement:** the native evaluators score via
> `invoke_async(prompt, structured_output_model=...)` — they need structured
> output through tool calling, not free text. A text-only chat-completions
> endpoint fails with `StructuredOutputException`. SAES enforces this with a
> **preflight probe** (`saes doctor --judge`) so a bad endpoint is rejected up
> front with an actionable message, never an opaque mid-run crash.

Verified judges include Bedrock (native + OpenAI-compatible), and — via the
Bedrock OpenAI API — DeepSeek, Kimi, and Qwen (see §10).

### 2.2 Framework-agnostic input

Any agent — any framework, any language — becomes evaluable simply by emitting
OTEL GenAI-convention traces to CloudWatch (or a local dump). SAES's ingestion
layer adapts to whatever spans each framework naturally emits and reconstructs a
uniform evaluation turn from them. **The adaptation lives in SAES ingestion, not
in the agent.** A bare `boto3` script with zero SAES-specific instrumentation
gets the same evaluator coverage as a native Strands agent (see §7 for how, and
§8 for the proof across four frameworks).

---

## 3. Architecture & implementation

### 3.1 What SAES owns vs. reuses

| Concern                                                                    | SAES-owned (this repo) | Native `strands-agents-evals`      |
| -------------------------------------------------------------------------- | ---------------------- | ---------------------------------- |
| Config schema / CLI                                                        | ✅ `config/`, `cli.py` | —                                  |
| Judge selection + probe                                                    | ✅ `judge/`            | model providers (`strands.models`) |
| Evaluator resolution (ids → native, custom LLM/code, trajectory)           | ✅ `evaluators/`       | the evaluator classes themselves   |
| Trace ingestion (factory, local reader, CloudWatch discovery + supplement) | ✅ `ingest/`           | providers + session mappers        |
| Run orchestration wiring + aggregation + gate                              | ✅ `run/`              | `Experiment.run_evaluations_async` |
| Reporting (JSON/HTML) + CloudWatch EMF sink                                | ✅ `report/`           | —                                  |
| Online worker (discover → complete → sample → score → emit)                | ✅ `online/`           | the scoring pipeline it drives     |
| Evaluators, templates, scoring, generation, detectors                      | —                      | ✅                                 |

### 3.2 Module map

```
src/saes/
├─ config/          # Pydantic config schema + YAML loader
│  ├─ schema.py     #   EvaluationConfig, JudgeModelConfig, DataSourceConfig,
│  │                #   SamplingConfig, SessionConfig, EvaluatorRef, sinks
│  └─ loader.py     #   load/parse, built-in id validation (derived from registry),
│                   #   secret redaction
├─ judge/           # LLM-as-a-Judge selection (differentiator #1)
│  ├─ providers.py  #   build_model(cfg) -> native strands Model (openai_compatible | bedrock)
│  ├─ probe.py      #   probe_judge() — structured-output capability preflight
│  ├─ structured.py #   tolerant JSON extraction + verdict parsing + repair loop
│  └─ base.py       #   Verdict, Judge protocol, TokenUsage
├─ ingest/          # framework-agnostic trace ingestion (differentiator #2)
│  ├─ source.py     #   load_sessions(cfg): otlp_file (local) via native mappers
│  ├─ cloudwatch.py #   CloudWatch: native provider + SAES session discovery + raw fetch
│  ├─ tool_supplement.py  # recover tool calls + conversation text from raw spans
│  ├─ cloudwatch_task.py  # supplemented task: synthesize turn + tool spans (F6/F10/F12)
│  └─ conformance.py#   saes doctor field-coverage report
├─ evaluators/      # resolve ids/custom to native Evaluators
│  ├─ registry.py   #   Builtin.* -> native class (+ judge injection, unique naming)
│  ├─ custom.py     #   custom LLM (OutputEvaluator) + custom code (@code_evaluator)
│  ├─ trajectory.py #   deterministic trajectory matchers (wrap native scorers)
│  └─ result.py     #   SAES EvaluationResult type
├─ run/             # on-demand orchestration
│  ├─ runner.py     #   run_on_demand(cfg): build cases+task, drive Experiment, aggregate
│  ├─ ground_truth.py#  JSONL dataset loading (expectedResponse/assertions/trajectory)
│  └─ gate.py       #   CI threshold rules -> pass/fail (exit code)
├─ report/          # outputs
│  ├─ build.py      #   flatten native report -> ReportDocument + rows
│  ├─ json_sink.py  #   JSON results
│  ├─ html_report.py#   self-contained HTML (Jinja2, judge-reasoning drill-down)
│  └─ cloudwatch_sink.py# EMF metrics + JSON log records to CloudWatch
├─ online/          # online evaluation
│  ├─ session_tracker.py#  span-quiescence completion + persisted scored-set
│  ├─ worker.py     #   cycle: discover→track→sample→rate-cap→score→emit
│  ├─ scoring.py    #   wires the worker to the native run pipeline + sink
│  └─ lambda_evaluator.py# code evaluator as a Lambda handler (AgentCore parity)
└─ cli.py           # eval | run | doctor | init | serve

cdk/                # optional infra: dashboard + alarms + least-privilege worker IAM
```

### 3.3 On-demand evaluation flow (`saes run`)

```
 config.yaml
     │  load_config (config/)
     ▼
 EvaluationConfig ──────────────────────────────────────────────┐
     │ data source                    judge                      │ evaluators
     ▼                                 ▼                          ▼
 ingest.load_sessions           judge.build_model         evaluators.resolve_evaluator
 (native mappers/providers)     (strands Model)           (native classes + judge)
     │  [native Session]s            │                          │
     └──────────────┬───────────────┴──────────────────────────┘
                    ▼
        run.run_on_demand  ── builds native Case per session (+ ground truth),
                                task(case) -> {output, trajectory: Session}
                    ▼
        strands_evals.Experiment.run_evaluations_async(task)   ← native engine
                    │  [EvaluationReport]
                    ▼
        run._aggregate  ── per-evaluator avg / pass_rate / n / errored
                    ▼
        report.build_report ── ReportDocument (+ rows w/ judge reasoning)
        ┌───────────┼─────────────┬───────────────────┐
        ▼           ▼             ▼                    ▼
   run.gate     json_sink     html_report      cloudwatch_sink
  (exit code)   results.json  report.html      EMF + JSON logs
```

Key wiring facts (verified against the real SDK):

- The native report's `detailed_results` is **evaluator-major and flattened** —
  one row per (evaluator, case). Aggregation keys off `report.cases[i]["evaluator"]`,
  which SAES sets to the AgentCore-style id.
- Native evaluators are **named by their id** so the same class can appear more
  than once in an Experiment (the engine rejects duplicate names).
- The judge is a native strands `Model`; SAES injects it via `model=`.

### 3.4 The ingestion seam (framework-agnostic)

`ingest.load_sessions(cfg)` returns native `Session` objects; SAES writes no
mapping code of its own for the happy path:

- **`otlp_file`** (offline/CI): read a local JSONL/OTLP dump, group spans by
  session id, hand each group to `detect_otel_mapper()`. Works for CloudWatch /
  OpenInference / LangChain-OTEL dict formats.
- **`cloudwatch`** (production): run a Logs Insights query to **discover session
  ids** (the native provider only reads by known id), then delegate per-session
  read+map to the native `CloudWatchProvider`. When the native mapper can't
  reconstruct a framework's spans, SAES's **supplement** (§7) fills the gap.
- **`live`**: native in-memory span capture for a running Strands agent.

`saes doctor --data-source` reports per-field coverage so gaps are visible
_before_ a run.

---

## 4. End-to-end usage: from building an agent to a scored report

This is the whole journey. Follow it top to bottom for a first working evaluation.

### 4.0 "I just cloned this repo and I have my own agent — where do I start?"

> **Want the single complete example, start to finish?** [WALKTHROUGH.md](WALKTHROUGH.md)
> is one linear path — clone → build an agent → deploy to AgentCore Runtime →
> traces to CloudWatch → SAES scores them — with no jumps and every command's
> real output. This section is the shorter "which path fits me" overview and the
> offline / try-it-now options.

You don't modify your agent and you don't touch SAES's source. SAES is a tool you
point at the traces your agent _already_ produces: install the CLI → get your
traces somewhere SAES can read → write one small YAML → run.

Every command below was run from a clean clone; the exact output is shown so you
know what "working" looks like.

#### Step 1 — Install (same for everyone)

**Prerequisite: Python 3.12** (the venv must use it — a system `python3` that is
3.9/3.10 will still install but 3.12 is what's verified). Check with `python3.12
--version` first; install it if missing (`sudo dnf install python3.12` /
`apt install python3.12` / `brew install python@3.12`).

```bash
git clone https://github.com/milan9527/CustomEval.git
cd CustomEval
python3.12 -m venv .venv
source .venv/bin/activate                 # ← activate FIRST; run everything below inside it
pip install --upgrade pip
pip install -e '.[dev]' openai            # installs the `saes` command + all deps
saes --help                               # ⇒ Commands: eval | run | doctor | init | serve
```

> If `pip install` fails with `No matching distribution found for
strands-agents`, you're not in the activated venv, or your pip points at a
> private index — activate first, or force public PyPI with
> `--index-url https://pypi.org/simple/`.

#### Step 2 — Prove it runs, using a trace sample bundled in the repo (~1 min)

Before wiring your own agent, confirm the whole chain works. The repo ships a real
trace fixture you can score immediately. You only need a judge — here, Amazon
Bedrock (uses your AWS credentials, no external API key):

```bash
pip install aws-bedrock-token-generator
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"

cat > try.yaml <<'YAML'
name: try-it
mode: on_demand
dataSource:
  type: otlp_file
  path: tests/fixtures/langgraph_session.jsonl   # ← a real sample that ships with the repo
judge:
  provider: openai_compatible
  model: "openai.gpt-oss-20b-1:0"
  base_url: "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
  api_key_env: SAES_JUDGE_API_KEY
evaluators: [Builtin.Helpfulness]
resultsSink:
  local: { html_report: ./out/report.html }
YAML

saes doctor --data-source tests/fixtures/langgraph_session.jsonl   # ⇒ OK — 1 session(s) reconstructed
saes doctor --judge try.yaml                                       # ⇒ ✓ structured output confirmed via tool calling
saes run -c try.yaml --json out/results.json --html out/report.html
```

Expected final output (verified):

```
try-it  (judge: openai.gpt-oss-20b-1:0)
  Builtin.Helpfulness              avg=0.833  pass=100%  n=1

JSON  → out/results.json
HTML  → out/report.html
```

If you see that, everything works and `out/report.html` has the judge's reasoning.
Now point it at your own agent.

> No AWS at all? Use any OpenAI-compatible endpoint that supports tool calling in
> the `judge` block instead (set `base_url`/`model` and put the key in
> `SAES_JUDGE_API_KEY`). A text-only endpoint won't work — `saes doctor --judge`
> tells you up front.

#### Step 3 — Point it at YOUR agent's traces (pick your path)

You do **not** need Strands and you do **not** add any SAES-specific telemetry —
SAES's ingestion adapts to whatever standard OTEL your framework emits (§7). The
only contract is "spans grouped by a `session.id`." Pick by what you have:

**Path A — your agent is on AgentCore Runtime** (traces auto-export to CloudWatch).
Just give `saes eval` the runtime id — it derives the log group, discovers the
sessions, and scores them. No YAML, no ground truth:

```bash
saes eval <your-runtime-id> --html out/report.html
#   scans the last 7 days by default; widen with --days 30 if the session is older
```

Options (mirroring AgentCore Evaluations):

```bash
saes eval --list-evaluators                          # show all built-in ids
saes eval <runtime> --all                            # all 13 built-ins (default: 12 reference-free)
saes eval <runtime> -e Builtin.Helpfulness,Builtin.Harmfulness   # choose evaluators
saes eval <runtime> --sampling 25                    # score 25% of sessions (deterministic)
saes eval <runtime> --judge-model gpt-4.1 --judge-base-url https://api.openai.com/v1
```

That's the whole thing for AgentCore. (For the three ground-truth evaluators —
Correctness / GoalSuccessRate / Trajectory\*Match — or a CI gate or custom
LLM/code evaluators, write a full config and use `saes run` / `saes serve` — §5,
§6, §11. To point at your own non-AgentCore CloudWatch log group, use a config
with `dataSource.type: cloudwatch` and `log_group_names`.)

**Path B — you can export a local OTEL/OTLP dump** (dev / CI, no trace store).
Save your spans to a JSONL file (one span record per line) and:

```yaml
dataSource:
  type: otlp_file
  path: ./my_traces.jsonl
```

```bash
saes doctor --data-source ./my_traces.jsonl      # ← ALWAYS run this first (see below)
saes run   -c try.yaml --html out/report.html
```

#### The one habit that saves you: `saes doctor` first

Before trusting scores, run `saes doctor --data-source <your dump>` (or, for
CloudWatch, confirm a session is discovered — §8.5 Step 1). It prints per-field
coverage and whether your traces reconstruct into **evaluable** sessions:

```
spans read: 4
field coverage:
  ✓ session id            4/4
  ✓ prompt / input        4/4
  ✓ completion / output   4/4
  ✗ tool name             0/4   (expected if this agent uses no tools)
OK — 2 session(s) reconstructed
```

A `✗` on session id / prompt / completion means your instrumentation is missing
those GenAI attributes — fix at the source, or you'll get empty scores (`n=0`).
Note: a Strands-scope **local dump** doesn't round-trip from a file (use the
CloudWatch source or in-memory for Strands); CloudWatch / OpenInference /
LangChain-OTEL dumps work from files. See §10 (F4).

The rest of §4 is the same journey in full detail; §5 is the config reference,
§8.5 is the exact pipeline each run executes.

### Step 1 — Have an agent that emits OTEL traces

SAES evaluates your agent from its OpenTelemetry traces; it never runs your
code. Your only job is to make the agent **emit OTEL GenAI-convention spans**,
grouped by a `session.id`. Three common situations:

- **Strands / AgentCore Runtime agent** — free. AgentCore's runtime is
  OTEL-instrumented and exports to CloudWatch automatically.
- **LangGraph / CrewAI / other framework** — enable its OpenTelemetry /
  OpenInference instrumentation; on AgentCore this exports automatically, or
  self-host an ADOT collector.
- **No framework at all** — a plain script's Bedrock calls are captured by
  AgentCore's botocore instrumentation; SAES reconstructs the turn from those
  standard spans (see §7). Zero SAES-specific code needed.

### Step 2 — Run your agent so traces exist

Exercise the agent on representative inputs. This produces the traces SAES scores
— either in a CloudWatch log group (production/online) or a local OTLP/JSONL dump
(offline/CI).

### Step 3 — Install SAES and verify your traces

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # activate FIRST
pip install -e '.[dev]' openai
saes doctor --data-source traces.jsonl                  # offline dump
```

`doctor` reports per-field coverage (session id, prompt/completion, tool name, …)
and whether sessions reconstruct. Fix any ✗ before continuing.

> The `openai` package is required whenever `judge.provider: openai_compatible`.
> For `judge.provider: bedrock` you only need AWS credentials. If `pip install`
> fails with `No matching distribution found for strands-agents`, you're likely
> not in the activated venv or your pip points at a private index — activate the
> venv, or force public PyPI with `--index-url https://pypi.org/simple/`.

### Step 4 — Pick a judge and verify it qualifies

```bash
export SAES_JUDGE_API_KEY=...              # or a Bedrock bearer token (§5.2)
saes doctor --judge eval.yaml              # → ✓ structured output confirmed
```

Text-only endpoints are rejected here, before a run.

### Step 5 — Write the config

```bash
saes init --agent-type rag --out eval.yaml   # scaffold with recommended evaluators
```

Then edit `eval.yaml`: point `dataSource` at your traces, set `judge`, choose
`evaluators`, and optionally add `groundTruth` and a CI `gate`. Full reference in §5.

### Step 6 — Evaluate

```bash
saes run -c eval.yaml --json out/results.json --html out/report.html
```

Console shows per-evaluator scores; the HTML report has per-result judge
reasoning. Exits non-zero if a `gate` threshold fails (wire into CI).

### Step 7 — (Optional) production monitoring

```bash
saes serve -c online.yaml --interval 60      # continuous; samples completed sessions
```

See §9.

### The whole loop, minimal

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]' openai
saes init --agent-type rag --out eval.yaml
# ...edit eval.yaml: dataSource.path, judge, evaluators...
export SAES_JUDGE_API_KEY=...
saes doctor --data-source traces.jsonl      # traces OK?
saes doctor --judge eval.yaml               # judge OK?
saes run -c eval.yaml --html out/report.html
```

---

## 5. Configuration reference

### 5.1 `eval.yaml`

```yaml
name: my-agent-quality
mode: on_demand # on_demand | online

dataSource:
  type: otlp_file # otlp_file (local) | cloudwatch
  path: ./traces.jsonl # for otlp_file
  # for cloudwatch instead:
  # type: cloudwatch
  # cloudwatch:
  #   log_group_names: ["/aws/bedrock-agentcore/runtimes/<agent>-DEFAULT"]
  #   agent_name: my-agent           # or discover the log group by name
  #   region: us-east-1
  #   lookback_days: 7

judge:
  provider: openai_compatible # openai_compatible | bedrock
  model: "gpt-4.1"
  base_url: "https://your-endpoint/v1" # required for openai_compatible
  api_key_env: "SAES_JUDGE_API_KEY" # env var NAME (never the key itself)
  params: { temperature: 0.0 }
  # provider: bedrock
  # model: "us.anthropic.claude-sonnet-4-5-..."   # no base_url/api_key needed

evaluators:
  - Builtin.Helpfulness
  - Builtin.Correctness # uses expectedResponse when present
  - Builtin.ToolSelectionAccuracy
  - Builtin.GoalSuccessRate # uses assertions
  - Builtin.TrajectoryInOrderMatch # deterministic; uses expectedTrajectory
  - id: hipaa_compliance # custom LLM evaluator
    type: llm
    level: trace
    instructions: |
      Score 1.0 if no PHI was disclosed without authorization, else 0.0.

groundTruth:
  path: ./ground_truth.jsonl # optional

gate: # optional CI thresholds
  - "Builtin.Helpfulness.avg >= 0.8"
  - "Builtin.Correctness.avg >= 0.9"
```

**Secrets:** the API key is read from the env var named by `api_key_env` and is
never stored on the model or serialized.

### 5.2 Amazon Bedrock as an OpenAI-compatible judge (verified)

```yaml
judge:
  provider: openai_compatible
  model: "openai.gpt-oss-20b-1:0" # or another Bedrock OpenAI model
  base_url: "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
  api_key_env: "SAES_JUDGE_API_KEY"
  params: { temperature: 0.0 }
```

Mint a short-term bearer token from ambient AWS credentials (inherits your IAM
role, auto-expires):

```bash
pip install aws-bedrock-token-generator
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
saes doctor --judge eval.yaml      # → ✓ structured output confirmed
```

> Alternatively use `provider: bedrock` (native AWS creds, no token). The
> OpenAI-compatible route is useful for one uniform config across providers.
> Both are verified end-to-end.

### 5.3 Ground truth (optional)

JSONL, one record per session, keyed by `sessionId`. Each evaluator reads only
the field it needs:

```json
{
  "sessionId": "s-123",
  "expectedResponse": "You have 40 hours of PTO.",
  "assertions": ["Agent retrieved the balance from the HR system"],
  "expectedTrajectory": ["lookup_employee", "get_pto_balance"]
}
```

- `expectedResponse` → `Builtin.Correctness`
- `assertions` → `Builtin.GoalSuccessRate`
- `expectedTrajectory` → `Builtin.Trajectory*Match`

---

## 6. The evaluator catalog

All evaluators are native `strands-agents-evals` classes under the hood, so
scores line up with managed AgentCore Evaluations.

| Evaluator                                                                   | Level   | Needs ground truth            | Kind          |
| --------------------------------------------------------------------------- | ------- | ----------------------------- | ------------- |
| `Builtin.GoalSuccessRate`                                                   | Session | `assertions` (optional)       | LLM           |
| `Builtin.Helpfulness`                                                       | Trace   | —                             | LLM           |
| `Builtin.Correctness`                                                       | Trace   | `expectedResponse` (optional) | LLM           |
| `Builtin.Coherence` / `Conciseness` / `Faithfulness`                        | Trace   | —                             | LLM           |
| `Builtin.Harmfulness` / `Refusal` / `Stereotyping`                          | Trace   | —                             | LLM           |
| `Builtin.InstructionFollowing` / `ResponseRelevance` / `ContextRelevance`\* | Trace   | —                             | LLM           |
| `Builtin.ToolSelectionAccuracy` / `ToolParameterAccuracy`                   | Tool    | —                             | LLM           |
| `Builtin.TrajectoryExactOrderMatch` / `InOrderMatch` / `AnyOrderMatch`      | Tool    | `expectedTrajectory`          | Deterministic |

\* `ContextRelevance` is aliased to ResponseRelevance in the current SDK (no
distinct native class in v1.0.2).

**Breakdown:** 12 pure LLM-as-judge (reference-free); `Correctness` +
`GoalSuccessRate` are LLM-judge _with optional_ ground truth; 3 trajectory
matchers are **deterministic** (no LLM, use `expectedTrajectory`). That's the "13
AgentCore built-ins + ContextRelevance alias + 3 trajectory scorers."

### Custom evaluators (AgentCore parity)

- **LLM** — `type: llm` + `instructions` (a rubric). Uses your judge; wraps
  native `OutputEvaluator`.
- **Code** — a deterministic function, referenced by `type: code`. One function
  body runs both locally and, in production, as a Lambda (M3):

```python
from saes.evaluators import code_evaluator, CodeVerdict

@code_evaluator(id="paystub_amount", level="trace")
def check(case) -> CodeVerdict:
    ok = "$8,333.33" in str(case.actual_output)
    return CodeVerdict(1.0 if ok else 0.0, "PASS" if ok else "FAIL")
```

---

## 7. Framework support: how any framework reaches full coverage

The central design goal: **any framework, or no framework, can be fully
evaluated** — the adaptation is in SAES ingestion, not in the agent. This section
explains the mechanism; §8 shows the proof.

### 7.1 The problem

The native `strands-agents-evals` mappers are tuned for the exact OTEL span shape
Strands emits (`AgentInvocationSpan` + `ToolExecutionSpan`). Other frameworks emit
different shapes:

- **LangGraph** — native read succeeds but yields only `InferenceSpan`s (no
  agent/tool spans).
- **No-framework / CrewAI** — the native mapper reconstructs _nothing_ and the
  read _raises_ `SessionNotFoundError`.

Left there, only Strands would get full coverage. But the data is present in all
cases — it's just in a shape the native mapper doesn't read.

### 7.2 The three SAES supplements

SAES's ingestion reconstructs a uniform evaluation turn from whatever standard
OTEL each framework emitted:

- **Tool supplement** (`ingest/tool_supplement`): recovers the tool trajectory
  (name / arguments / result) from raw Converse `toolUse`/`toolResult` spans,
  bridging `trace_id → session.id` (botocore tool spans carry no session id; an
  OpenInference span in the same trace does).
- **Role-aware turn recovery** (`_iter_role_texts`): recovers the user prompt and
  the **final answer** from the roled botocore Bedrock spans — `body.message`
  (role=assistant, `finish_reason=end_turn`) and `body.{input,output}.messages`.
  _The final answer is captured by AgentCore's botocore instrumentation; the fix
  was reading it correctly, not changing the agent._
- **Turn + tool-span synthesis** (`cloudwatch_task.supplement_turns`): builds
  native `AgentInvocationSpan`s (with `available_tools`) and `ToolExecutionSpan`s
  from the recovered turns — the exact shapes the native `TraceExtractor` consumes
  at TRACE / SESSION / TOOL level. This is what lets the two tool-level LLM
  evaluators run for non-Strands agents.
- **Per-turn reconstruction** (`tool_supplement._reconstruct_turns`): for a
  _multi-turn_ session it groups recovered text + tools by `trace_id` (one
  AgentCore trace = one turn), orders turns by time, and synthesizes **one
  `AgentInvocationSpan` per turn** — so each turn's prompt is paired with that
  turn's own answer, not a mixed last-answer. Without this, a 3-turn session
  mispairs (e.g. turn-3 "Paris?" with a turn-1 "Tokyo" answer) and scores wrong.

All supplements are best-effort and never raise into a run; they apply
automatically in both `saes run` and `saes serve`.

### 7.3 Result: uniform coverage

With the supplements in place, **all 15 built-in evaluators run for all four
frameworks.** The remaining differences are _score_ differences driven by each
agent's actual behavior — which is exactly what an evaluation suite should
surface. One fidelity note: for non-Strands agents, synthesized `available_tools`
carry tool _names_ only (raw spans don't include tool descriptions / JSON
schemas), so the tool-level evaluators reason over names + observed calls rather
than full tool specs.

### 7.4 What YOUR agent must emit — the OTEL contract, by framework

You do not write mapping code, but your traces must carry a few things for SAES
(and the native `strands-agents-evals` mappers underneath) to reconstruct an
evaluable session. This is the checklist to follow when building an agent in each
framework. **Always verify with `saes doctor --data-source <dump>` before you
rely on the scores** — it prints exactly which of these fields are present.

> **Building agents with Claude Code / Claude Agent SDK?** This whole section
> is also packaged as an agent skill at
> [`.claude/skills/otel-eval-contract/`](.claude/skills/otel-eval-contract/SKILL.md).
> With the repo open in Claude Code the skill loads automatically whenever you
> write agent code or instrumentation, so generated agents follow this contract
> without you restating it; it also works as a system-prompt include for any
> other agent-generating pipeline. The contract below is the requirement —
> the skill is just the convenient way to apply it. It was used to produce
> `examples/agents/claudesdk_tools/`, which evaluated 7/7 on the first deploy.

#### The universal contract (every framework)

1. **A `session.id`** on the spans (accepted keys: `session.id`,
   `gen_ai.session.id`, or `session_id`). This is how spans group into a
   conversation. On AgentCore Runtime you get it by passing `--session-id` to
   `agentcore invoke`; reused across turns → a multi-turn session.
2. **Prompt/input text** — any of: `gen_ai.prompt` (or indexed
   `gen_ai.prompt.N.content`), `gen_ai.input.messages`, `input.value`,
   `llm.input_messages.*`, or `traceloop.entity.input`.
3. **Completion/output text** — any of: `gen_ai.completion` (or
   `gen_ai.completion.N.content`), `gen_ai.output.messages`, `output.value`,
   `llm.output_messages.*`, or `traceloop.entity.output`.
4. **A `scope.name`** on each span — this selects the mapper (see below).
5. **`traceId` + `spanId`** — standard OTEL; one **trace per turn** (SAES groups
   multi-turn sessions by trace and orders by span time).

Minimum to reconstruct _anything_: `session.id` **plus** prompt **or** completion.
Tool-level and trace-level evaluators need more (below).

#### The scope name decides which mapper runs

The native mapper is chosen by each span's `scope.name`. Only three values are
recognized natively:

| `scope.name`                                                                       | Native mapper                             | Typical source                               |
| ---------------------------------------------------------------------------------- | ----------------------------------------- | -------------------------------------------- |
| `strands.telemetry.tracer`                                                         | Strands mapper (full: agent + tool spans) | Strands SDK                                  |
| `opentelemetry.instrumentation.langchain`                                          | LangChain-OTEL mapper                     | LangChain/LangGraph via OTEL instrumentor    |
| `openinference.instrumentation.langchain`                                          | OpenInference mapper                      | OpenInference LangChain instrumentor         |
| anything else (`…crewai`, `botocore…`, `com.anthropic.claude_code.events`, custom) | **none matches** → native read may raise  | CrewAI, bare boto3, Claude Agent SDK, custom |

**If your scope isn't one of the three, you are not broken** — SAES's supplement
(§7.2) recovers the trajectory + turns from the raw Bedrock Converse spans
(`botocore` `toolUse`/`toolResult`, roled `body.message`). You just rely on the
supplement rather than the native mapper. That is exactly how CrewAI and the
no-framework agent reach full coverage.

#### To unlock each evaluator level

- **Trace-level** (Helpfulness, Correctness, Coherence, …) needs a reconstructed
  **turn**: a user prompt + the agent's **final answer**. Emit both (contract
  items 2–3). Strands emits an `AgentInvocationSpan` natively; for other
  frameworks SAES synthesizes it from the recovered prompt+answer.
- **Tool-level** (ToolSelectionAccuracy, ToolParameterAccuracy) needs the
  **tool call** — name, arguments, and result. Strands emits a
  `ToolExecutionSpan`; other frameworks just need their Bedrock `toolUse`/
  `toolResult` Converse blocks in the spans (SAES synthesizes the tool span). The
  arguments must be the real tool input for ToolParameterAccuracy to be meaningful.
- **Trajectory match** (deterministic) needs the ordered tool-call names, which
  come from the same `toolUse` blocks — plus an `expectedTrajectory` in ground
  truth.

#### Per-framework notes (from real deployments)

- **Strands** — nothing to do. Native OTEL emits `AgentInvocationSpan` +
  `ToolExecutionSpan` + `InferenceSpan`; all 15 evaluators work, multi-turn
  included. The reference path.
- **LangGraph** — enable OpenInference (`LangChainInstrumentor().instrument()`)
  or the OTEL LangChain instrumentor so `scope.name` is one of the two LangChain
  values. Tool calls flow through the Bedrock Converse spans → SAES recovers
  them. (With `ChatBedrockConverse` on an inference-profile ARN, set
  `provider="anthropic"`.)
- **CrewAI** — its scope is `openinference.instrumentation.crewai`, which the
  native mapper does **not** match, so the native read raises — expected. SAES
  recovers the trajectory + answer from the Converse spans. Current gap: CrewAI's
  per-turn _user prompt_ isn't always in the shape the recovery reads, so
  ResponseRelevance can miss on it (GoalSuccessRate / tools still work).
- **No framework (bare boto3)** — no instrumentation to add: AgentCore's botocore
  Bedrock instrumentation already captures the Converse request/response
  (including the final answer as `body.message`). SAES reconstructs the turn +
  tools from those. Just make sure your Bedrock calls go through the instrumented
  client (they do by default on AgentCore Runtime).
- **Claude Agent SDK** (`claude-agent-sdk`) — the SDK drives Bedrock through a
  bundled CLI **subprocess**, so AgentCore's botocore instrumentation never sees
  the model calls and nothing is captured for free. Two verified ways to make it
  evaluable:
  - **Path A — the SDK's built-in OpenTelemetry (OTLP → collector).**
    Set `CLAUDE_CODE_ENABLE_TELEMETRY=1` + `OTEL_LOG_RAW_API_BODIES=1` and export
    over OTLP to a collector that forwards to CloudWatch (it does **not** write to
    CloudWatch directly — its docs also warn against the `console` exporter). Its
    record shape is unlike the others: one scope
    (`com.anthropic.claude_code.events`) for everything, the event kind in
    `attributes["event.name"]`, turns keyed by `attributes["prompt.id"]` (no
    `traceId`), and clean text in dedicated `user_prompt` / `assistant_response`
    events. SAES ingestion handles all of this; the CLI's internal session-title
    call (`query_source="generate_session_title"`) is filtered so the scored answer
    is the real turn. **Verified end to end** (agent → OTLP → ADOT collector →
    CloudWatch → `saes eval`, all 12 reference-free evaluators scored) — see
    [examples/agent_sdk/](../examples/agent_sdk/README.md). Emit-side gap: the SDK's
    `tool_result` event carries only metadata, so the tool _result_ payload isn't
    recoverable (trajectory, args, prompt, and final answer are).
  - **Path B — emit the contract by hand (no collector needed).** The agent
    itself emits the §7.4 contract with the OTEL SDK (verified end-to-end;
    source: `examples/agents/claudesdk_tools/`): one root span per invocation
    carrying `session.id` (from `RequestContext.session_id`) + `gen_ai.prompt` +
    `gen_ai.completion`; one OTEL event per turn with roled
    `body.{input,output}.messages` (the shape role-aware recovery reads); and
    per tool call, Converse-shaped `toolUse`/`toolResult` event bodies with the
    real arguments + result (what the tool supplement recovers) — so unlike
    Path A, tool _results_ ARE recoverable. Two gotchas: SDK tool callbacks run
    in the SDK's own asyncio tasks, so their spans must be parented on the root
    span via an explicit `SpanContext` (contextvars don't cross the CLI
    transport); and the container needs `ca-certificates` + `ripgrep` + a
    writable `HOME` for the bundled CLI, with `CLAUDE_CODE_USE_BEDROCK=1` in
    the SDK's `options.env`. Evaluated on a real deployment: 7/7 configured
    evaluators ran on a 3-turn session, tool-level 4/4 calls at 1.0, per-turn
    prompt/answer/tool pairing correct.
- **Raw Anthropic API SDK** (`anthropic` / `AnthropicBedrock`) — **not evaluable
  as-is.** It calls Bedrock over its own httpx + SigV4 client, which botocore's
  instrumentation never sees, so it emits no usable OTEL. The fix is on the emit
  side (instrument its HTTP client, or route through boto3). This is the negative
  counterpart to the Claude _Agent_ SDK above — the contract is on the telemetry,
  not the SDK name.

#### The one habit

```bash
saes doctor --data-source your_dump.jsonl
```

A `✓` on session id + prompt + completion means sessions reconstruct; a `✗` tells
you exactly which attribute your instrumentation is missing — fix it at the source
before trusting scores. (§4.0 has a sample of the output.)

Two refinements from the Claude Agent SDK run:

- **Reading doctor output on supplement-path agents** (Claude SDK, CrewAI,
  no-framework): the prompt/completion _field-coverage_ rows may show `✗` when
  the content lives in event **bodies** rather than span attributes — the field
  rows check attributes, the supplement reads bodies. That is fine as long as
  the final "sessions reconstructed" line succeeds (exit 0).
- **Smoke-test locally before deploying**: run one invocation under an
  in-memory OTEL exporter, dump the records as JSONL, and doctor that — it
  catches contract violations without a ~10-minute deploy cycle. Working
  example: `examples/agents/claudesdk_tools/smoke_local.py`.

---

## 8. Evaluation scenarios & results analysis

Three complementary bodies of evidence, all with a real Bedrock judge.

### 8.1 Every built-in discriminates good from bad

Each built-in was given a purpose-built GOOD scenario and a BAD scenario
(hand-built multi-turn native Sessions, judge-only). An evaluator that scores
everything high is useless — this is the counter-check that they actually
discriminate.

| Evaluator             | GOOD  | BAD   | GOOD scenario → BAD scenario                                        |
| --------------------- | ----- | ----- | ------------------------------------------------------------------- |
| Helpfulness           | 0.833 | 0.167 | clear password-reset steps → vague deflection ×2                    |
| Correctness           | 1.000 | 0.000 | "capital of France is Paris" → "…is Berlin"                         |
| Coherence             | 1.000 | 0.000 | consistent Sunday hours → "open every day" then "closed Sundays"    |
| Conciseness           | 1.000 | 0.000 | "4." → rambling paragraph for "2+2"                                 |
| Faithfulness          | 1.000 | 0.000 | recalls "red shirt" → later says "blue jacket"                      |
| Harmfulness           | 1.000 | 0.000 | professional advice → "spread rumors and sabotage"                  |
| InstructionFollowing  | 1.000 | 0.000 | "Blue." to one-word request → long answer ignoring the rule         |
| ResponseRelevance     | 1.000 | 0.000 | answers the question → off-topic blurb                              |
| ContextRelevance      | 1.000 | 0.000 | budget summary for budget Q → cookie recipe                         |
| Refusal               | 1.000 | 0.000 | refuses bomb request → provides instructions                        |
| Stereotyping          | 1.000 | 0.000 | neutral description → "only young men can do it"                    |
| GoalSuccessRate       | 1.000 | 0.000 | books flight + confirmation → never books                           |
| ToolSelectionAccuracy | 1.000 | 0.000 | `get_weather` for weather Q → `calculate` for weather Q             |
| ToolParameterAccuracy | 1.000 | 0.000 | `get_weather(city=Tokyo)` → `get_weather(city=Paris)` for a Tokyo Q |

**14/14 discriminate good > bad.** The judge gives specific reasons, e.g.
wrong-fact: _"The correct answer is Paris, not Berlin. Berlin is the capital of
Germany…"_

A deliberately-bad agent was also **deployed on AgentCore** and scored through the
full online path (auto-OTEL → CloudWatch → `saes serve` → judge → results):
Helpfulness **0.0**, InstructionFollowing **0.0** — versus the good agent's
0.833 on the same path. Discrimination holds both in isolation and end-to-end.

### 8.2 The cross-framework scenario

Four agents, each in a different framework, all deployed on AgentCore Runtime,
all exposing the **same two tools** and asked the **same questions** — so
tool-level evaluators and trajectory matchers compare apples to apples:

```
Tools:  get_weather(city) -> forecast    calculate(expression) -> arithmetic
Prompts: "What's the weather in Tokyo?"       -> get_weather(Tokyo)
         "What is 15% of 240?"                -> calculate(...)
         "Weather in Paris, and what is 12*8?" -> both tools
```

### 8.3 Four frameworks × 15 evaluators — the matrix

Real AgentCore CloudWatch traces, real Bedrock OpenAI-compatible judge
(`openai.gpt-oss-20b-1:0`), via `saes serve`'s supplemented CloudWatch task.
The grid below is a **verbatim re-run** (`framework_matrix.py`, saved to
`FRAMEWORK_MATRIX_OUTPUT.txt`):

| Evaluator               | strands   | noframe   | langgraph | crewai    |
| ----------------------- | --------- | --------- | --------- | --------- |
| Helpfulness             | 0.833     | 0.667     | 0.833     | 0.833     |
| Correctness             | 1.000     | 1.000     | 1.000     | 1.000     |
| Coherence               | 1.000     | 1.000     | 1.000     | 1.000     |
| Conciseness             | 1.000     | 1.000     | 1.000     | 0.500\*   |
| Faithfulness            | 1.000     | 1.000     | 1.000     | 1.000     |
| Harmfulness             | 1.000     | 1.000     | 1.000     | 1.000     |
| InstructionFollowing    | 1.000     | 1.000     | 1.000     | 1.000     |
| ResponseRelevance       | 1.000     | 1.000     | 1.000     | 1.000     |
| ContextRelevance        | 1.000     | 1.000     | 1.000     | 1.000     |
| Refusal                 | 0.000\*\* | 0.000\*\* | 0.000\*\* | 0.000\*\* |
| Stereotyping            | 1.000     | 1.000     | 1.000     | 1.000     |
| GoalSuccessRate         | 1.000     | 1.000     | 0.000\*   | 0.000\*   |
| ToolSelectionAccuracy   | (ran†)    | 1.000     | 1.000     | 0.000\*   |
| ToolParameterAccuracy   | (ran†)    | 1.000     | 1.000     | 0.000\*   |
| TrajectoryAnyOrderMatch | 1.000     | 0.500     | 1.000     | 1.000     |
| **Evaluators that RAN** | **15/15** | **15/15** | **15/15** | **15/15** |

\* **Content** outcomes, not pipeline gaps: the evaluator _ran_; it scored low
because the terse ground truth didn't match the fuller answer, or the session
accumulated many drifting turns. The table's point is _which evaluators run_.
Individual content scores vary run-to-run with the judge (e.g. Helpfulness for a
given framework may land 0.667 or 0.833 on different runs); the structure — all
15 running for all four frameworks — is stable.
\*\* Refusal=0.0 on benign traffic is expected polarity: these agents never had
anything to refuse.
† Strands's two tool-level cells show blank in the matrix script only because of
a transient judge error on that pass, caught per-cell. Strands **does** have
native `ToolExecutionSpan`s (verified this run: `session_has_tool_spans=True`),
and the CLI path (§11 Step 3a) scored its `ToolParameterAccuracy=1.0` across all
10 tool calls — so this is run noise, not a gap.

### 8.4 Analysis — how each framework reaches 15/15

- **Strands** — native OTEL emits `AgentInvocationSpan` + `ToolExecutionSpan`
  directly. No supplement needed. The reference case.
- **LangGraph** — native read yields `InferenceSpan`s. The **turn supplement**
  synthesizes the agent span; the **tool supplement** recovers tool calls and
  synthesizes `ToolExecutionSpan`s.
- **No-framework / CrewAI** — native read _raises_. SAES substitutes an empty
  Session, then: the tool supplement recovers the trajectory, role-aware recovery
  lifts the final answer, and turn+tool-span synthesis builds the native spans the
  extractor needs.

**Before the supplements, no-framework and CrewAI could only run 1/15
(trajectory). After, they run 15/15.** This is the concrete proof that the
framework-agnostic claim holds at full depth — achieved entirely in SAES
ingestion, with no agent change and no redeploy (verified against the same
already-deployed agents' CloudWatch data).

#### A fifth framework, end to end: the Claude Agent SDK

Beyond the four AgentCore agents above, the **Claude Agent SDK**
(`claude-agent-sdk`) was run through the _full emit path_ — agent → OTLP → an ADOT
collector → CloudWatch → `saes eval` — to prove the OTLP→CloudWatch wiring, not
just the SAES-side parsing. Over 3 real sessions all 12 reference-free evaluators
produced scores, with a faithfully reconstructed turn (prompt, final answer,
`["Bash"]` trajectory, args). Full write-up, sources, and the browsable HTML report:
**[examples/agent_sdk/](../examples/agent_sdk/README.md)**.

**The full pipeline — from agent run to evaluation result.** There are two opposite
directions here; keeping them straight is the whole mental model:

```
┌─ EMIT (runtime) ────────────────────────┐   ┌─ EVALUATE (offline, SAES side) ─────────┐
│  Claude Agent SDK        (run_agent.py)  │   │  saes eval /aws/saes/agentsdk-results    │
│    │  built-in OpenTelemetry             │   │    1. discover session ids in the group  │
│    │  emits claude_code.* events         │   │    2. fetch raw span records             │
│    ▼  OTLP/http → :4318                  │   │    3. reconstruct the turn(s)            │
│  ADOT collector (Docker)                 │   │       (prompt · final answer · tools)    │
│    │  awscloudwatchlogs exporter         │   │    4. score each turn with an LLM judge  │
│    ▼                                     │   │    5. write report to LOCAL disk         │
│  CloudWatch Logs ────────────────────────────▶   out/agentsdk_report.{html,json}       │
│  /aws/saes/agentsdk-results              │   │       (the OUTPUT — NOT in CloudWatch)   │
│  (raw telemetry ONLY — the INPUT)        │   │                                          │
└──────────────────────────────────────────┘   └──────────────────────────────────────────┘
```

1. **Agent runs** — with `CLAUDE_CODE_ENABLE_TELEMETRY=1` the SDK's built-in OTEL
   emits each step (user prompt, API bodies, tool decision/result, final answer) as
   `claude_code.*` events over **OTLP**. It does not write CloudWatch directly.
2. **Collector forwards** — an ADOT collector receives OTLP and its
   `awscloudwatchlogs` exporter writes each raw event into
   `/aws/saes/agentsdk-results`.
3. **CloudWatch holds the raw telemetry** — this is the evaluation _input_, not a
   result.
4. **SAES evaluates** — `saes eval` reads that group back, reconstructs the turn, and
   scores it with the judge.
5. **Result lands on local disk** — the terminal table plus any `--html`/`--json`
   files. `saes eval` does **not** write results back to CloudWatch.

**Why the report isn't inside `/aws/saes/agentsdk-results`.** That log group is the
agent's _input_, not SAES's _output_ — they flow in opposite directions. `saes eval`
is an offline reader: it pulls raw records _out_ of CloudWatch, scores locally, and
writes the report to `out/`. (Analogy: the log group is the camera _tape_; the report
is the _write-up_ you make after watching it — the write-up never appears on the
tape.) To land the _scores_ in CloudWatch, use online mode, which writes them to a
**separate** results log group:

```bash
saes serve /aws/saes/agentsdk-results --results-log-group /aws/saes/agentsdk-scores
```

Its CloudWatch record shape is unlike the AgentCore agents' (single scope, event kind
in `attributes["event.name"]`, turns keyed by `prompt.id`, internal session-title
calls filtered) — SAES ingestion handles it; see §7.4. Honest limits: the SDK exports
no tool-_result_ payload, and the tool-parameter judge is non-deterministic on this
borderline case. The raw `anthropic` API SDK, by contrast, emits no usable OTEL at all
(bypasses botocore) — the negative boundary that sharpens the "contract is on the
telemetry" claim.

### 8.5 The evaluation, step by step

This is exactly what happens when you evaluate one framework's agent — the
concrete pipeline behind every column of the §8.3 matrix. It's the same sequence
whether you run it ad hoc (`framework_matrix.py`), on-demand (`saes run`), or
online (`saes serve`); only the trigger differs. **For the actual copy-paste
commands to deploy and evaluate each framework, see [§11 "The four-framework
matrix"](#the-four-framework-matrix-deploy--evaluate-on-agentcore--concrete-commands).**
The steps below explain what those commands do internally.

#### Step 0 — Deploy the agent (once, per framework)

Each agent exposes the same two tools (`get_weather`, `calculate`) and is
deployed to AgentCore Runtime. AgentCore ships `aws-opentelemetry-distro`, so
**traces auto-export to CloudWatch** at
`/aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT` — no telemetry code in the
agent. What each framework _emits_ differs, and that difference is the whole
story:

| Framework    | Instrumentation source                           | Spans that reach CloudWatch                                      |
| ------------ | ------------------------------------------------ | ---------------------------------------------------------------- |
| Strands      | native Strands OTEL tracer                       | `AgentInvocationSpan`, `InferenceSpan`, `ToolExecutionSpan`      |
| LangGraph    | OpenInference (`LangChainInstrumentor`)          | LangChain spans → map to `InferenceSpan`s (no tool/agent spans)  |
| No-framework | AgentCore's **botocore** Bedrock instrumentation | raw Converse spans: `toolUse`/`toolResult`, roled `body.message` |
| CrewAI       | OpenInference (`.crewai` scope)                  | raw Converse spans (scope unknown to native mapper)              |

#### Step 1 — Discover the session id from CloudWatch

The native `CloudWatchProvider` can only read _by a known session id_, so SAES
owns discovery: a Logs Insights query for distinct `attributes.session.id` in the
lookback window.

```python
from saes.config.schema import CloudWatchSource
from saes.ingest.cloudwatch import build_provider, discover_session_ids

cfg = CloudWatchSource(
    log_group_names=["/aws/bedrock-agentcore/runtimes/saesnoframe-6AXcAT2oW4-DEFAULT"],
    region="us-east-1", lookback_days=1)
provider = build_provider(cfg)
session_id = discover_session_ids(provider, cfg)[0]
# -> '60bb9061-3630-49eb-825f-6ebaf38a5b19'
```

> Allow ~90–100s after invoking the agent: CloudWatch trace delivery + Logs
> Insights indexing have a lag the online worker's poll interval absorbs.

#### Step 2 — Build the supplemented task

`build_supplemented_task` wraps the native `provider.as_task()`. When invoked for
a session it runs the native read, then applies the supplements as needed. The
native read behaves differently per framework — this is the branch point:

```python
from saes.ingest.cloudwatch_task import build_supplemented_task
task = build_supplemented_task(provider, cfg)
out = task(case)              # case.input == session_id
session = out["trajectory"]   # a native Session, ready for the extractor
```

What happens inside, per framework (real reconstructed span types shown):

- **Strands** — native read succeeds with
  `['AgentInvocationSpan', 'InferenceSpan', 'ToolExecutionSpan']`. `need_tools`
  and `need_turn` are both false → **no supplement runs**. The Session is used
  as-is.
- **LangGraph** — native read succeeds with `['AgentInvocationSpan',
'InferenceSpan']` but _no tool spans_. `need_tools` is true → SAES fetches the
  raw spans, recovers the tool trajectory, and synthesizes `ToolExecutionSpan`s.
- **No-framework / CrewAI** — native read **raises** `SessionNotFoundError`
  (reconstructed span types `[]`). SAES catches it, substitutes an empty Session,
  then runs the full supplement (tool trajectory + role-aware turn + tool spans).

The mapper's per-span WARNING spam during the native read is quieted to a single
INFO summary line (F8), e.g.
`session 60bb9061: recovered 4-step tool trajectory via supplement (native read had failed)`.

#### Step 3 — Reconstruct the evaluation turn (the supplement)

For any framework whose native read didn't already produce the needed spans, SAES
reconstructs them from the raw CloudWatch records (`fetch_session_records` →
`extract_session_tool_calls`):

1. **Bridge trace_id → session_id.** botocore tool spans carry no `session.id`;
   an OpenInference span in the same `trace_id` does. SAES links them.
2. **Recover the tool trajectory.** Pull `toolUse` (name + arguments) and
   `toolResult` (content) from the raw Converse spans, order-independent.
3. **Recover the turn text, role-aware.** Read the user prompt and the **final
   answer** from roled spans — `body.message` (role=assistant,
   finish_reason=end_turn) and `body.{input,output}.messages`. (This is the fix
   from F11: the final answer was always in CloudWatch; SAES just had to read it
   from the right field instead of guessing by string length.)
4. **Synthesize native spans.** Build an `AgentInvocationSpan` (user_prompt +
   agent_response + `available_tools`) and one `ToolExecutionSpan` per recovered
   tool call — the exact shapes the native `TraceExtractor` consumes at TRACE /
   SESSION / TOOL level.

After this, every framework's Session contains the same evaluable structure,
regardless of what it originally emitted.

#### Step 4 — Attach ground truth (only some evaluators need it)

The `Case` carries optional ground truth; each evaluator reads only its field:

```python
from strands_evals import Case
case = Case(name=session_id, input=session_id, session_id=session_id,
            expected_output="The weather in Tokyo is 22C; ...",   # → Correctness
            expected_assertion="Answered weather and math using tools.",  # → GoalSuccessRate
            expected_trajectory=["get_weather", "calculate"])     # → Trajectory*Match
```

The 12 reference-free LLM evaluators need none of this.

#### Step 5 — Resolve evaluators and inject the judge

```python
from saes.evaluators import resolve_evaluator
from saes.config.schema import EvaluatorRef, JudgeModelConfig
from saes.judge.providers import build_model

judge = build_model(JudgeModelConfig(
    provider="openai_compatible", model="openai.gpt-oss-20b-1:0",
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
    api_key_env="SAES_JUDGE_API_KEY"))

ev = resolve_evaluator(EvaluatorRef(id="Builtin.ToolSelectionAccuracy",
                                    type="builtin"), judge)
```

`resolve_evaluator` maps the id to the native evaluator class, names the instance
by its id (so duplicates can coexist in one Experiment), and injects the judge.
Trajectory ids resolve to the deterministic matcher instead (no judge).

#### Step 6 — Run the evaluator over the reconstructed Session

```python
from strands_evals import Experiment
report = await Experiment(cases=[case], evaluators=[ev]).run_evaluations_async(task)
out = report.detailed_results[0][0]
print(out.score, out.reason)
# 1.0  "The user explicitly asked for the weather in Tokyo. The provided tool, get_weather..."
```

Internally the native `TraceExtractor` walks the Session at the evaluator's level:

- **TRACE level** (Helpfulness, Correctness, …) — reads the `AgentInvocationSpan`'s
  user_prompt + agent_response.
- **SESSION level** (GoalSuccessRate) — reads the whole conversation.
- **TOOL level** (ToolSelectionAccuracy, ToolParameterAccuracy) — reads each
  `ToolExecutionSpan` (tool_call name/args, tool_result) + `available_tools`.

Because Step 3 produced all three span kinds for every framework, every level of
evaluator has data — that's why the matrix is 15/15 across the board.

#### Step 7 — Aggregate, gate, and emit

Per-evaluator `avg / pass% / n`; optional CI `gate` (non-zero exit on failure);
results to JSON/HTML and/or CloudWatch EMF + JSON logs (§9). In the online worker
this is the tail of every cycle; the session is then marked scored so it's never
re-scored.

#### The whole thing, condensed

```
discover session id  ─┐
                      ▼
build_supplemented_task ── native read ──► [raises?] ── empty Session
                      │                          │
                      ▼                          ▼
              need_tools/need_turn? ── yes ──► fetch raw spans
                      │                          │  bridge trace→session
                      │                          │  recover trajectory + roled turn text
                      │                          ▼
                      │                   synthesize AgentInvocationSpan
                      │                   + ToolExecutionSpan(s) + available_tools
                      ▼                          │
              uniform native Session  ◄──────────┘
                      │
       resolve evaluators (+judge)   attach ground truth
                      │
                      ▼
     Experiment.run_evaluations_async(task)  ──►  scores + judge reasoning
                      │
        aggregate → gate → JSON/HTML/CloudWatch
```

---

## 9. Online / production evaluation

Continuous evaluation of a live agent's traffic. Like `saes eval`, the zero-config
path is **just the runtime id** — no YAML:

```bash
saes serve myagent-XyZ123                     # continuous; polls every 60s
saes serve myagent-XyZ123 --once              # one cycle (CI/cron)
saes serve myagent-XyZ123 --sampling 5 --session-timeout 30 --all
```

It derives the runtime's log group, uses the 12 reference-free evaluators by
default, and **auto-creates a results sink** at `/aws/saes/<runtime>-results`
(override with `--results-log-group`). Same evaluator flags as `saes eval`
(`-e`, `--all`, `--sampling`, `--judge-model`, …), plus:

- `--session-timeout N` — a session is "complete" after N minutes with no new span.
- `--interval N` — seconds between polling cycles (default 60).
- `--state FILE` — persist which sessions were scored across restarts.
- `--print-scores` — also print each scored batch's per-evaluator scores in the
  terminal (they always go to CloudWatch either way; online mode is
  results-to-CloudWatch by default, unlike `saes eval` which prints).

Verified live: `saes serve saesstrands-... --once` scored 3/3 sessions and wrote
results to the auto-derived CloudWatch group.

### What the worker does each cycle

1. **Discover** session ids in the lookback window (Logs Insights).
2. **Detect completion** via span-quiescence — no new span for
   `session.timeout_minutes` ⇒ the session is complete (matching managed
   AgentCore's approach).
3. **Sample** per `SamplingConfig` with a rolling-window rate cap.
4. **Score** via the same pipeline as `saes run` (including the supplements).
5. **Emit** EMF metrics + JSON result records to CloudWatch. Each session is
   scored at most once (persisted scored-set); failures retry next cycle.

### Full-config alternative (`--config`)

For CI gates, custom LLM/code evaluators, a non-AgentCore log group, or a rolling
rate cap, pass a YAML instead of a runtime id:

```yaml
mode: online
dataSource:
  type: cloudwatch
  cloudwatch:
    log_group_names: ["/aws/bedrock-agentcore/runtimes/<agent>-DEFAULT"]
    region: us-east-1
judge: { provider: bedrock, model: "..." }
evaluators: [Builtin.Helpfulness, Builtin.Correctness]
session: { timeout_minutes: 30 } # ~ your agent's typical session length
sampling: { percentage: 5.0, max_per_minute: 100 }
resultsSink:
  cloudwatch:
    {
      log_group: "/aws/saes/online-results",
      metrics_namespace: "SAES/Evaluations",
    }
```

```bash
saes serve -c online.yaml --interval 60 --state state.json   # continuous loop
saes serve -c online.yaml --once                             # one cycle (CI/cron)
```

Custom code evaluators can also run as a Lambda (`online/lambda_evaluator.py`),
and `cdk/` provisions a dashboard + alarms + least-privilege worker role.

### Relationship to managed AgentCore Evaluations

Same shape (CloudWatch OTEL spans → group into sessions → sample → score → EMF
metrics + JSON logs), self-hosted, with a bring-your-own OpenAI-compatible judge
and the `SAES/Evaluations` metrics namespace instead of the managed one.

---

## 10. Verification log: what was proven, and the bugs found

Everything below was run with a **real LLM judge**, not stubs. The demand for
real end-to-end runs is what surfaced the bugs.

### Proven end-to-end

- **Offline pipeline** — real Strands agent → OTEL spans → native Session → real
  Bedrock judge → scores with reasoning (Helpfulness 0.833, Correctness 1.0,
  Coherence 1.0).
- **Online pipeline** — a real agent deployed on **AgentCore Runtime**,
  auto-exporting OTEL to CloudWatch; `saes serve --once` discovered the quiescent
  session, scored it, and wrote EMF metrics + JSON results back to CloudWatch.
- **BYO judge** — the `openai_compatible` path verified against **Bedrock's
  OpenAI API** (`openai.gpt-oss-20b-1:0`), plus **DeepSeek, Kimi, and Qwen** via
  the same endpoint. All qualify (support tool calling) and score end-to-end.
- **Judge comparability caveat, demonstrated:** Qwen rated Helpfulness 1.0 vs.
  0.833 for DeepSeek/Kimi on the _same_ trace — scores are only comparable when
  the judge is held constant. SAES stamps `judgeModel` on every result.
- **Framework-agnostic** — Strands, LangGraph, CrewAI, and a no-framework
  `boto3` script all evaluated from real AgentCore CloudWatch traces; all reach
  15/15 evaluators (§8).

### Bugs found and fixed (the informative ones)

| #    | Bug                                                                                                       | Fix                                                                                                                                |
| ---- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| F1   | `_final_output(session)` guessed non-existent Session attrs → blank output                                | read `AgentInvocationSpan.agent_response`                                                                                          |
| F3   | native evaluators need structured output; text-only endpoints crash mid-run                               | preflight probe (`saes doctor --judge`) rejects them up front                                                                      |
| F7   | `doctor` field-coverage false-negatives on traceloop/indexed keys                                         | prefix-wildcard aliases (`gen_ai.prompt.*`, `traceloop.entity.*`, …)                                                               |
| F8   | native-mapper WARNING spam (~19 lines) on successful non-Strands runs                                     | quiet those loggers during the read; emit one INFO summary                                                                         |
| F9   | supplement skipped when native read **raises** — scored empty, exactly when most needed                   | catch the raise, substitute empty Session, still supplement                                                                        |
| F10  | non-Strands LLM evaluators returned None (no agent span)                                                  | synthesize `AgentInvocationSpan` from recovered turn text                                                                          |
| F11→ | **wrongly** concluded the final answer "wasn't in CloudWatch" and needed an agent-side fix                | it _was_ there (botocore captured it); fixed **role-aware** extraction in SAES ingestion — lifted no-framework/CrewAI 1/15 → 13/15 |
| F12  | the last 2 tool-level LLM evaluators still couldn't run for non-Strands                                   | synthesize native `ToolExecutionSpan`s + `available_tools` from recovered tool calls → 13/15 → 15/15                               |
| F13  | non-Strands **multi-turn** sessions mispaired turns (turn-3 prompt with turn-1 answer) → wrong 0.0 scores | reconstruct one turn per `trace_id`, time-ordered; one `AgentInvocationSpan` per turn (verified: LangGraph 3-turn 0.0→0.833/1.0…)  |
| F14  | mapper WARNING spam leaked on multi-**session** eval despite quieting                                     | the per-task quieter raced across concurrent `to_thread` tasks; made it refcounted + lock-guarded                                  |

> **Lesson from F9/F11:** `scored 1/1` (a session was processed) is not the same
> as a non-zero score, and "the data isn't there" is a claim to verify against
> raw spans, not assume. Recording the actual per-framework numbers — and
> inspecting real captured spans — is what caught both.

### Known limitations

- Local `otlp_file` dumps of **Strands-scope** spans don't round-trip (the
  in-memory mapper wants `ReadableSpan` objects, not dicts). Use the `live`
  in-memory path or the CloudWatch source for Strands; dict-format
  CloudWatch/OpenInference/LangChain dumps work fine from files.
- Synthesized `available_tools` for non-Strands agents carry tool names only (raw
  spans lack tool descriptions/JSON schemas) — a fidelity note, not a missing
  evaluator.

---

## 11. Reproduce

### Unit tests

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]' openai
pytest -q                    # 199 passing
ruff check src/ tests/       # clean
```

### The good/bad discrimination suite (judge-only, no deploy)

```bash
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
python builtin_suite.py      # ~26 judge calls; every built-in good>bad
python bad_examples.py       # multi-turn bad sessions score 0.0
```

### The four-framework matrix (deploy + evaluate on AgentCore) — concrete commands

This is the full sequence to reproduce §8.3, with the exact commands. The agent
sources live in `agents/{strands,noframework,langgraph,crewai}_tools/` in the
verification workspace (`/home/ec2-user/saes_run`); adapt paths for your checkout.

> **Verified end-to-end** against the four deployed runtimes: both the matrix
> script (Step 3b, printed the full §8.3 grid) and the per-framework CLI path
> (Step 3a, all four returned `scored N/N` and wrote real scores to CloudWatch —
> Strands/noframe/langgraph tool-level 1.0, CrewAI trajectory 1.0). The exact
> outputs are inline below.

#### Prereqs

```bash
source .venv/bin/activate    # SAES installed
pip install bedrock-agentcore bedrock-agentcore-starter-toolkit \
            langgraph langchain-aws crewai crewai-tools \
            openinference-instrumentation-langchain openinference-instrumentation-crewai
export BEDROCK_MODEL_ID="us.anthropic.claude-sonnet-4-5-20250929-v1:0"   # the agent's model
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
```

#### Step 1 — Deploy each framework's agent to AgentCore (~5 min each, CodeBuild)

Same procedure per framework; only the directory and name change. All four expose
the same `get_weather` + `calculate` tools:

```bash
cd agents/strands_tools          # then noframework_tools / langgraph_tools / crewai_tools
export AGENTCORE_SUPPRESS_RECOMMENDATION=1
printf '\n\n\n\n\n' | agentcore configure -e agent.py -n saesstrands -rf requirements.txt --create
sed -i 's/ecr_auto_create: false/ecr_auto_create: true/' .bedrock_agentcore.yaml
agentcore deploy -env BEDROCK_MODEL_ID="$BEDROCK_MODEL_ID"
```

Gotchas (already handled in the provided agent sources): the starter toolkit may
not write a `Dockerfile` (one is provided per agent); `ecr_auto_create` defaults
to false (the `sed` above flips it); LangGraph's `ChatBedrockConverse` with an
inference-profile ARN needs `provider="anthropic"`.

#### Step 2 — Invoke each agent (produces real OTEL traces in CloudWatch)

```bash
cd agents/<fw>_tools
agentcore invoke '{"prompt": "What is the weather in Tokyo?"}'
agentcore invoke '{"prompt": "What is 15% of 240?"}'
agentcore invoke '{"prompt": "Weather in Paris, and what is 12*8?"}'
```

Wait ~90–100s for trace delivery + Logs Insights indexing before evaluating.

#### Step 3a — Evaluate one framework via the CLI (`saes serve`)

This is what a user actually types. Write a config per framework — only the log
group changes — then run one online cycle. **Verified working today against all
four deployed runtimes** (each returned `scored N/N`, exit 0).

First find the session id (needed for the trajectory ground truth). Sessions age
out of the lookback window, so set `lookback_days` to cover your session — today
the deployed sessions were 44–63h old, so `lookback_days: 3`:

```bash
python -c "
from saes.config.schema import CloudWatchSource
from saes.ingest.cloudwatch import build_provider, discover_session_ids
cfg = CloudWatchSource(log_group_names=['/aws/bedrock-agentcore/runtimes/saesnoframe-6AXcAT2oW4-DEFAULT'], region='us-east-1', lookback_days=3)
print(discover_session_ids(build_provider(cfg), cfg))"
# -> ['d8d24446-a5c7-4523-b8f8-dc53a2cfc401', ...]
```

Then write the config + ground truth and run one cycle:

```bash
cat > eval-noframe.yaml <<'YAML'
name: eval-noframe
mode: online
dataSource:
  type: cloudwatch
  cloudwatch:
    log_group_names: ["/aws/bedrock-agentcore/runtimes/saesnoframe-6AXcAT2oW4-DEFAULT"]
    region: us-east-1
    lookback_days: 3                # cover your session's age
judge:
  provider: openai_compatible
  model: "openai.gpt-oss-20b-1:0"
  base_url: "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
  api_key_env: SAES_JUDGE_API_KEY
evaluators:
  - Builtin.Helpfulness
  - Builtin.ToolSelectionAccuracy   # ← runs for non-Strands too, via the supplement (§7)
  - Builtin.ToolParameterAccuracy
  - Builtin.TrajectoryAnyOrderMatch
session:  {timeout_minutes: 1}      # a session with no span for 1 min counts as complete
sampling: {percentage: 100.0}
groundTruth: {path: ./gt_noframe.jsonl}      # for the trajectory matcher
resultsSink:
  cloudwatch: {log_group: "/aws/saes/fw-results", metrics_namespace: "SAES/Fw", dimensions: [agentId, evaluatorId]}
YAML

echo '{"sessionId": "d8d24446-a5c7-4523-b8f8-dc53a2cfc401", "expectedTrajectory": ["get_weather", "calculate"]}' > gt_noframe.jsonl

saes serve -c eval-noframe.yaml --once
#   serving online eval for 'eval-noframe' (timeout=1.0m, sampling=100.0%)
#     scored 2/2 session(s) this cycle
#   cycle: ready=2 scored=2 deferred=0 errored=0
```

> A session with many tool calls makes many judge calls at tool level. The
> LangGraph session (13 tool calls) took several minutes — allow a generous
> timeout when you invoke `saes serve`.

Read the scores back from the sink:

```bash
python -c "
import boto3, json
logs = boto3.client('logs', region_name='us-east-1')
s = logs.describe_log_streams(logGroupName='/aws/saes/fw-results', orderBy='LastEventTime', descending=True, limit=1)['logStreams'][0]['logStreamName']
for e in logs.get_log_events(logGroupName='/aws/saes/fw-results', logStreamName=s, startFromHead=False, limit=50)['events']:
    m = json.loads(e['message'])
    if m.get('type') == 'saes.result': print(m['evaluatorId'], m['score'])"
```

Real scores read back today (per framework, abbreviated):

```
strands   : Helpfulness 0.833 | ToolParameterAccuracy 1.0 (×10 tool calls) | TrajectoryAnyOrderMatch 1.0
noframe   : Helpfulness 0.833 | ToolSelectionAccuracy 1.0 | ToolParameterAccuracy 1.0 | Trajectory 0.5
langgraph : Helpfulness 0.833 | ToolSelection/ToolParameter across 13 calls | Trajectory 1.0
crewai    : Helpfulness 0.833 | ToolParameterAccuracy (ran; 0.0 on its 8-call session) | Trajectory 1.0
```

Repeat for the other three frameworks by changing `log_group_names` and the
session id (`saesstrands-ZhPiI77pEM-DEFAULT` / `saeslanggraph-vSzHF7G235-DEFAULT`
/ `saescrewai-JjA6Jp5dHw-DEFAULT`).

#### Step 3b — All four × all 15 evaluators in one table (the matrix script)

To reproduce the §8.3 table directly (evaluates every deployed framework against
every built-in and prints the grid), use the workspace script — its judge is set
via `BEDROCK_MODEL_ID`:

```bash
cd /home/ec2-user/saes_run
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
export BEDROCK_MODEL_ID="openai.gpt-oss-20b-1:0"
python framework_matrix.py       # 4 frameworks × 15 evaluators over real CloudWatch traces (~10 min)
```

This was run today and printed the full §8.3 grid (saved to
`FRAMEWORK_MATRIX_OUTPUT.txt`). Two things to match to your deployment: the
runtime ids hard-coded near the top (`AGENTS = {...}`) and `lookback_days`
(currently `3`, since today's sessions were ~2 days old — widen it or invoke the
agents fresh so discovery finds the sessions). The verbose variant
(`framework_matrix_verbose.py`) additionally prints reconstructed span types +
judge reasoning per evaluator.

#### Step 4 — Cleanup

```bash
for d in strands_tools noframework_tools langgraph_tools crewai_tools; do
  (cd agents/$d && AGENTCORE_SUPPRESS_RECOMMENDATION=1 agentcore destroy)
done
aws logs delete-log-group --log-group-name /aws/saes/fw-results --region us-east-1
```

---

## 12. Troubleshooting

| Symptom                                              | Cause / fix                                                                                                                                          |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `StructuredOutputException` mid-run                  | Judge endpoint lacks tool calling. Run `saes doctor --judge`; pick a qualifying endpoint.                                                            |
| `ModuleNotFoundError: openai`                        | `pip install openai` (needed for `openai_compatible`).                                                                                               |
| `saes doctor` shows ✗ for session id / prompt        | Instrumentation missing those GenAI attributes — fix at the source.                                                                                  |
| 0 sessions / empty scores from a local dump          | Strands-scope dumps don't round-trip from file; use in-memory or CloudWatch. Dict-format dumps work.                                                 |
| An evaluator returns nothing for a non-Strands agent | Usually the data _is_ in CloudWatch in a different span shape; the supplements handle the known ones. Inspect raw spans before assuming it's absent. |
| Scores shifted between runs                          | Judge model changed. Hold the judge constant; SAES stamps `judgeModel` on every result.                                                              |
| Gate exit code always 0                              | Ensure `gate:` rules are in the config; exit is non-zero only when a rule fails.                                                                     |
