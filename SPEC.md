# Strands Agent Evaluation Suite (SAES) — Technical Specification

**Status:** Draft v0.1
**Date:** 2026-07-10
**Audience:** Engineers, solution architects, and open-source contributors building agent evaluation tooling on the Strands Agents SDK.

---

## 1. Overview

**Strands Agent Evaluation Suite (SAES)** is an open-source evaluation solution for AI agents built with the [Strands Agents SDK](https://strandsagents.com/). It provides a self-hostable alternative and complement to [Amazon Bedrock AgentCore Evaluations](https://aws.amazon.com/blogs/machine-learning/build-reliable-ai-agents-with-amazon-bedrock-agentcore-evaluations/), integrated natively with [Amazon Bedrock AgentCore Observability](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html).

The core differentiator: **any OpenAI-compatible endpoint that supports tool calling / structured output can serve as the LLM-as-a-Judge**, so teams are not locked into the managed judge models used by AgentCore Evaluations. Users bring their own judge — OpenAI, Azure OpenAI, a self-hosted vLLM server (with guided decoding), LiteLLM proxy, Amazon Bedrock, or any conformant server.

> **Judge requirement (verified against `strands-agents-evals` v1.0.2):** the native evaluators score via `invoke_async(prompt, structured_output_model=...)`, i.e. they require the judge to emit **structured output through tool calling**, not plain text. A bare chat-completions endpoint that only returns free text is **not** sufficient and fails with `StructuredOutputException`. SAES verifies this capability with a probe on `saes doctor` / `saes init` (§3.5) and documents it prominently rather than letting runs fail opaquely. Most real endpoints (OpenAI, Azure, Bedrock, vLLM with guided decoding) qualify; some minimal local servers (older Ollama/LM Studio configs) do not.

**Framework-agnostic input contract:** SAES does not require the evaluated agent to be built with Strands. **Any third-party agent — in any language or framework — becomes evaluable simply by emitting OpenTelemetry traces (GenAI semantic conventions) to CloudWatch.** The agent name "Strands" reflects the SDK SAES is *built with* (evaluation engine + judge layer), not a constraint on what it can *evaluate*. If the traces conform to the OTEL GenAI conventions and land in a CloudWatch log group, SAES evaluates them.

**Design stance:** *reuse over rebuild.* SAES wraps the existing open-source [`strands-agents-evals`](https://strandsagents.com/docs/user-guide/evals-sdk/quickstart/index.md) package for the evaluation engine (built-in evaluators, trajectory scorers, experiment generation, simulators, detectors) and adds only the three things that package lacks: (1) OpenAI-compatible judge selection as a first-class config surface, (2) a CloudWatch/OTEL trace-ingestion adapter that is agent-framework-agnostic, and (3) an online sampling worker + CloudWatch results emission. Minimal net-new code.

### 1.1 Goals

- **G1** — Provide AgentCore-Evaluations-equivalent capabilities (built-in evaluators, ground truth, session/trace/tool-level scoring) as open source, runnable locally, in CI, or in-account.
- **G2** — Let users select any LLM model via an OpenAI-compatible API (with tool-calling / structured-output support) as the judge for LLM-as-a-Judge evaluators.
- **G3** — Consume agent behavior from AgentCore Observability (OpenTelemetry traces stored in CloudWatch) so the same telemetry powers both operational monitoring and quality evaluation — no separate instrumentation.
- **G3a (framework-agnostic ingestion)** — Evaluate **any** agent, regardless of framework or language, provided it emits OTEL GenAI-convention traces to a CloudWatch log group. The only contract is the wire format, not the SDK.
- **G3b (reuse)** — Build on `strands-agents-evals` for the evaluation engine; keep SAES-specific code limited to judge selection, ingestion adapter, and the online worker.
- **G4** — Support both **offline/on-demand** (development, CI/CD, regression, ground-truth datasets) and **online/continuous** (production sampling) evaluation modes.
- **G5** — Write evaluation results back to CloudWatch as metrics and structured logs so scores appear alongside operational signals in the AgentCore Observability dashboard.

### 1.2 Non-Goals

- Not a replacement for AgentCore Observability's telemetry pipeline; SAES consumes it, it does not re-implement trace collection.
- Not a managed service — no control plane, no multi-tenant hosting. It is a library + CLI + optional worker.
- Not an agent-authoring framework; it evaluates agents, it does not build them.
- Not limited to Strands-built agents. SAES is *implemented with* the Strands Evals SDK but *evaluates* any agent whose traces reach CloudWatch in OTEL GenAI format (LangGraph, custom Python/TS/Java/Go agents, OpenInference-instrumented apps, etc.).
- Not rebuilding an evaluation engine from scratch — the engine is `strands-agents-evals`; SAES is the integration + judge-selection layer around it.

### 1.3 Relationship to existing tooling

| Capability | AgentCore Evaluations (managed) | Strands Evals SDK (OSS) | **SAES (this spec)** |
|---|---|---|---|
| Judge model | Managed, fixed built-in models | Bedrock default (Claude), configurable `Model` | **Any OpenAI-compatible endpoint + Bedrock + Strands providers** |
| Built-in evaluators | 13, fixed config | Library of evaluators | 13 built-in evaluators reimplemented as open, editable prompt templates |
| Ground truth | expectedResponse / assertions / expectedTrajectory | Per-`Case` expected_output/trajectory | Same three modes, unified schema |
| Online (production) eval | Yes, CloudWatch sampling | No | **Yes — trace sampler + worker** |
| Observability backend | CloudWatch (managed) | CloudWatch/Langfuse remote providers | **AgentCore Observability (CloudWatch), pluggable** |
| Hosting | AWS-managed | Self-run | Self-run (local / container / Lambda / ECS) |

SAES is built **on top of** the open-source `strands-agents-evals` package where possible and adds: OpenAI-compatible judge selection as a first-class configuration surface, an AgentCore Observability trace ingestion adapter, an online-evaluation worker, and CloudWatch results emission.

---

## 2. Concepts & Data Model

SAES adopts the same three-level hierarchy AgentCore Evaluations uses, because it maps directly onto OpenTelemetry GenAI semantic conventions:

- **Session** — a complete conversation (all traces sharing a `session.id`).
- **Trace** — one user turn: request → response, containing all steps taken.
- **Span** — a single operation within a trace (LLM call, tool invocation, retrieval).

Evaluators operate at one of these levels.

### 2.1 Core objects

```
EvaluationConfig
 ├─ id, name, description
 ├─ mode: "on_demand" | "online"
 ├─ dataSource: DataSourceConfig
 ├─ evaluators: [EvaluatorRef]          # up to N; built-in or custom
 ├─ judge: JudgeModelConfig             # default judge; per-evaluator override allowed
 ├─ sampling: SamplingConfig            # online mode only
 ├─ groundTruth: GroundTruthRef | null
 └─ resultsSink: ResultsSinkConfig
```

```
GroundTruth (per session or per trace, keyed by sessionId / traceId)
 ├─ expectedResponse: string | null            # trace level  → Correctness
 ├─ assertions: [string] | null                # session level → GoalSuccessRate
 └─ expectedTrajectory: [toolName] | null       # session level → Trajectory* matchers
```

```
EvaluationResult
 ├─ configId, evaluatorId, level, sessionId, traceId?, spanId?
 ├─ score: float                        # normalized 0.0–1.0
 ├─ label: string | null                # e.g. PASS/FAIL or ordinal bucket
 ├─ reason: string                      # judge's reasoning (LLM) or explanation (code)
 ├─ judgeModel: string                  # provenance
 ├─ groundTruthUsed: bool
 ├─ ignoredReferenceInputFields: [string]
 └─ timestamp, latencyMs, tokenUsage
```

### 2.2 Score normalization

All evaluators emit a normalized `score` in `[0.0, 1.0]`. Ordinal scales (e.g. 1–5, 7-level helpfulness) are linearly mapped; binary evaluators emit `0.0` or `1.0`. Raw scores are preserved in a `rawScore` field for audit.

---

## 3. Judge Model Selection (OpenAI-Compatible API)

This is the defining feature. SAES exposes a single `JudgeModelConfig` that resolves to a judge model client. LLM-as-a-Judge evaluators use this client to score.

### 3.1 Configuration schema

```yaml
judge:
  provider: openai_compatible   # openai_compatible | bedrock | strands
  model: "gpt-4.1"              # model id passed to the endpoint
  base_url: "https://my-vllm.internal:8000/v1"   # OpenAI-compatible endpoint
  api_key_env: "SAES_JUDGE_API_KEY"              # env var name; never inline the key
  params:
    temperature: 0.0
    max_tokens: 1024
    top_p: 1.0
  timeout_s: 60
  max_retries: 3
  # optional: structured-output enforcement mode
  structured_output: "json_schema"   # json_schema | tool_call | prompt
```

### 3.2 Provider resolution

SAES reuses Strands Agents' model provider layer so the judge is configured exactly like any Strands agent model:

- **`openai_compatible`** → `strands.models.openai.OpenAIModel` with
  `client_args={"api_key": <resolved>, "base_url": <base_url>}`. This covers OpenAI, Azure OpenAI (with the right base_url), vLLM (with guided decoding), LiteLLM proxy, SageMaker OpenAI-compatible endpoints, and any conformant server **that supports tool calling / structured output** (see the judge requirement in §1 and the probe in §3.5). Text-only endpoints are rejected up front by the probe rather than failing mid-run.
- **`bedrock`** → `strands.models.BedrockModel` (default judge = Claude on Bedrock, matching the Strands Evals default). Uses AWS credentials, no API key.
- **`strands`** → any registered Strands `Model` provider by name (Anthropic, Mistral, Cohere, etc.), for teams already standardized on a provider SDK.

The judge model is dependency-injected into evaluators. A single evaluation run may use one default judge with **per-evaluator overrides** (e.g. a stronger model for `Correctness`, a cheap local model for `Refusal`).

### 3.3 Structured-output contract

LLM judges must return a deterministic, parseable verdict. SAES enforces a fixed output schema and always requests **reasoning before score** (matching AgentCore's convention of a `reason` field preceding a `score` field, which improves calibration):

```json
{
  "reason": "string — the judge's justification",
  "score": 3,
  "label": "optional string"
}
```

Enforcement strategy is negotiated by capability, in this priority order:
1. **`json_schema`** — response_format with JSON schema (OpenAI, vLLM guided decoding).
2. **`tool_call`** — force a single tool/function call whose arguments are the verdict.
3. **`prompt`** — instruction-only fallback with a robust JSON extractor + one repair retry.

The chosen strategy is auto-detected per endpoint with an explicit override in config. Parse failures trigger up to `max_retries` re-asks before the result is marked `errored` (never silently dropped).

### 3.4 Judge-agnostic prompting

Built-in evaluator templates are the AgentCore-published prompts (§4), which already carry their own output JSON schema and instruct "return only pure JSON." The structured-output layer wraps this: on `json_schema`/`tool_call`-capable endpoints it enforces the template's schema natively; on prompt-only endpoints it relies on the template's built-in JSON instruction plus the extractor/repair path. The verdict enum defined in each template is parsed and mapped to the normalized score (§4.0). Because these prompts were tuned on a specific managed judge, cross-judge score comparability carries the usual caveat below.

> **Caveat / calibration note:** Scores are only comparable across runs when the judge model is held constant. Switching judge models (or even model versions) can shift score distributions. SAES stamps `judgeModel` on every result and **warns** when a saved baseline is compared against results produced by a different judge. Teams should re-baseline after changing judges.

### 3.5 Judge capability probe

Because the native evaluators require structured output via tool calling (§1), a judge endpoint that only returns free text will fail every evaluation with `StructuredOutputException`. To catch this **before** a run rather than mid-flight, SAES probes the configured judge:

- `saes doctor --judge` (and the tail of `saes init`) send a tiny structured-output request through the resolved judge model and confirm a parseable structured result comes back.
- **Pass:** proceed. **Fail:** report clearly that the endpoint lacks tool-calling/structured-output support, name the endpoint, and point to the requirement — no opaque mid-run failure.
- The probe is cheap (one tiny call) and never runs implicitly during scoring; it's an explicit preflight.

This makes the requirement a checked contract, not a footnote.

---

## 4. Built-in Evaluators

SAES exposes the **13 built-in evaluators** matching AgentCore Evaluations by mapping each AgentCore-style id (`Builtin.Helpfulness`, …) onto the **native `strands-agents-evals` evaluator classes** (`HelpfulnessEvaluator`, `CorrectnessEvaluator`, `ToolSelectionAccuracyEvaluator`, …). SAES does **not** reimplement evaluators or author prompt templates.

**Why native, not ported templates:** `strands-agents-evals` ships all 13 evaluators *and* their prompt templates (in `strands_evals.evaluators.prompt_templates`), which are the AgentCore-equivalent prompts with versioned scoring (`version="v0"`). Reusing them directly means SAES tracks the SDK's templates and scoring for free, honoring the reuse decision (D1). Every native evaluator accepts `model=`, so SAES injects the selected OpenAI-compatible judge uniformly. (An earlier draft hand-ported the AWS-published prompt text; that was rejected as duplicate work once the native templates were verified to be equivalent and versioned.)

### 4.0 Resolution (one mapping table, native classes)

The registry maps ids to native classes and injects the judge model:

```python
BUILTIN_EVALUATORS = {
    "Builtin.Helpfulness": HelpfulnessEvaluator,
    "Builtin.Correctness": CorrectnessEvaluator,
    # ... all 13 ...
}
def resolve_evaluator(ref, judge_model):
    return BUILTIN_EVALUATORS[ref.id](model=judge_model)
```

The native evaluators own placeholder assembly, prompt rendering, judge invocation, structured output, and score normalization — SAES adds none of it. Adding/removing a built-in is a one-line table change.

### 4.1 Catalog

The native evaluators carry the per-level placeholder contracts and scoring scales internally (Session → `context`/`available_tools`; Trace → `context`/`assistant_turn`; Tool → `available_tools`/`context`/`tool_turn`). Scores are normalized to `[0,1]` by the SDK.

| Evaluator id | Native class | Level | Ground truth field |
|---|---|---|---|
| `Builtin.GoalSuccessRate` | `GoalSuccessRateEvaluator` | Session | `assertions` |
| `Builtin.Helpfulness` | `HelpfulnessEvaluator` | Trace | — |
| `Builtin.Correctness` | `CorrectnessEvaluator` | Trace | `expectedResponse` (optional) |
| `Builtin.Coherence` | `CoherenceEvaluator` | Trace | — |
| `Builtin.Conciseness` | `ConcisenessEvaluator` | Trace | — |
| `Builtin.Faithfulness` | `FaithfulnessEvaluator` | Trace | — |
| `Builtin.Harmfulness` | `HarmfulnessEvaluator` | Trace | — |
| `Builtin.InstructionFollowing` | `InstructionFollowingEvaluator` | Trace | — |
| `Builtin.ResponseRelevance` | `ResponseRelevanceEvaluator` | Trace | — |
| `Builtin.ContextRelevance` | `ResponseRelevanceEvaluator` ⚠️ aliased | Trace | — |
| `Builtin.Refusal` | `RefusalEvaluator` | Trace | — |
| `Builtin.Stereotyping` | `StereotypingEvaluator` | Trace | — |
| `Builtin.ToolSelectionAccuracy` | `ToolSelectionAccuracyEvaluator` | Tool | — |
| `Builtin.ToolParameterAccuracy` | `ToolParameterAccuracyEvaluator` | Tool | — |

> ⚠️ **ContextRelevance is aliased.** `strands-agents-evals` v1.0.2 exposes no distinct `ContextRelevanceEvaluator`; SAES maps `Builtin.ContextRelevance` to `ResponseRelevanceEvaluator` as the closest native match and surfaces this via a `CONTEXT_RELEVANCE_IS_ALIASED` flag. Revisit when the SDK adds a dedicated evaluator.
| `ToolParameterAccuracy` | Tool | binary/ordinal → 0…1 | — |

> Note: the "13 built-in" count follows AgentCore's grouping (session=1, trace=10, tool=2), plus the ContextRelevance alias. Scoring scales and normalization are owned by the native evaluators. The trajectory matchers below are ground-truth **scorers** attached to trajectory ground truth rather than standalone LLM evaluators.

### 4.2 Ground-truth trajectory scorers (deterministic, no LLM)

Selectable as built-in ids; each consumes `expectedTrajectory` and compares it
against the agent's actual tool sequence:

- `Builtin.TrajectoryExactOrderMatch` — same tools, same order, no extras.
- `Builtin.TrajectoryInOrderMatch` — expected tools appear in order; extras allowed between.
- `Builtin.TrajectoryAnyOrderMatch` — all expected tools present, any order; extras allowed.

**Implementation (verified against `strands-agents-evals` v1.0.2):** the SDK ships these as scorer *functions* (`exact_match_scorer` / `in_order_match_scorer` / `any_order_match_scorer`) that its LLM `TrajectoryEvaluator` calls internally — they are **not** standalone deterministic evaluators. SAES wraps them directly as native `Evaluator` subclasses (`TrajectoryMatchEvaluator`), so they score without any LLM: the actual tool-name sequence is extracted from the native `Session` (`ToolExecutionSpan.tool_call.name`), the expected sequence comes from the `Case.expected_trajectory`, and the native matcher function returns the `[0,1]` score. When `expectedTrajectory` is absent, the evaluator reports `N/A` rather than a misleading score.

### 4.3 Evaluator interaction guidance (documentation baked into the tool)

The suite surfaces the AgentCore-documented distinctions to prevent misreading, e.g.:
- `Correctness` (factual) vs. `Faithfulness` (consistent with sources — can be faithful to flawed input yet wrong).
- `Helpfulness` (advances goal) vs. `ResponseRelevance` (addresses the question).
- `ToolParameterAccuracy` is only meaningful when `ToolSelectionAccuracy` is high — SAES emits a diagnostic hint when parameter scores are read on low-selection sessions.

Recommended starting set (3–4 evaluators aligned to agent purpose) is emitted by `saes init` based on an interactive agent-type prompt (customer-service, RAG, tool-heavy).

### 4.4 Provenance & versioning

Because built-ins are native `strands-agents-evals` evaluators, prompt/scoring provenance is the SDK's evaluator `version` (e.g. `"v0"`). SAES stamps the resolved evaluator name and the `strands-agents-evals` package version on every result so runs are reproducible and comparable. Score comparability across runs still requires holding the judge model constant (§3.4).

---

## 5. Ground Truth

SAES supports the three independent, optional ground-truth reference inputs, consumed only by the evaluators that need them (all others fall back to reference-free mode):

| Field | Type | Scope | Consumed by |
|---|---|---|---|
| `expectedResponse` | string | Trace (keyed by `traceId`) | `Correctness` |
| `assertions` | list[string] | Session | `GoalSuccessRate` |
| `expectedTrajectory` | list[toolName] | Session | Trajectory matchers |

Rules:
- All three may be supplied together; each evaluator reads only its field. Fields it cannot use are reported in `ignoredReferenceInputFields`.
- Reference-free evaluators (`Helpfulness`, `ResponseRelevance`, …) run in the same batch and ignore ground truth entirely.
- Custom evaluators may reference ground-truth fields via `{expectedResponse}`, `{assertions}`, `{expectedTrajectory}` placeholders in their instructions.

### 5.1 Dataset format

Ground-truth datasets are JSONL, one record per evaluation unit:

```json
{"sessionId": "s-123", "traceId": "t-1", "input": "What is my PTO balance?",
 "expectedResponse": "You have 40 hours of PTO.",
 "assertions": ["Agent retrieved the balance from the HR system",
                "Agent did not disclose another employee's data"],
 "expectedTrajectory": ["lookup_employee", "get_pto_balance"]}
```

Datasets can be authored by hand, exported from production traces (promote-to-dataset flow — a failing production trace becomes a regression case), or generated (see §9).

---

## 6. Custom Evaluators

Two extension mechanisms mirroring AgentCore's model, each built on a native `strands-agents-evals` primitive — SAES adds only the config/registration glue.

### 6.1 Custom LLM evaluators — native `OutputEvaluator`

The user supplies a rubric (natural-language instructions + scoring guidance); SAES injects the selected OpenAI-compatible judge. This wraps the native `OutputEvaluator(rubric=..., model=judge)` directly.

```yaml
- id: hipaa_compliance
  type: llm
  level: trace                    # session | trace | tool
  judge_override:                 # optional; else uses the default judge (§3)
    provider: openai_compatible
    model: "gpt-4.1"
  instructions: |                 # the rubric passed to OutputEvaluator
    Evaluate whether the agent disclosed PHI without authorization.
    Score 1.0 if no PHI was disclosed; 0.0 if any PHI was disclosed.
```

`OutputEvaluator` handles prompt assembly (input/output/expected-output/environment-state), judge invocation, and structured scoring. It also supports `uses_environment_state` for asserting side effects (DB rows, file contents) — available to custom evaluators for free.

### 6.2 Custom code evaluators — native `Evaluator` subclass

Deterministic Python callables for checks that don't need language understanding — exact value verification, format/schema compliance, business-rule enforcement, high-volume low-cost scoring. Registered with a decorator; SAES adapts the callable to the native `Evaluator` interface (`evaluate`/`evaluate_async → list[EvaluationOutput]`).

```python
from saes.evaluators import code_evaluator, CodeVerdict

@code_evaluator(id="paystub_amount", level="trace")
def check_paystub(case) -> CodeVerdict:
    ok = "$8,333.33" in str(case.actual_output)
    return CodeVerdict(score=1.0 if ok else 0.0,
                       label="PASS" if ok else "FAIL",
                       reason="Verbatim paystub amount present" if ok
                              else "Expected amount not found verbatim")
```

The callable receives a native `EvaluationData` (`input`, `actual_output`, `expected_output`, `actual_trajectory`, `metadata`, …). Local runs invoke it directly through the native pipeline. Online runs (M3) deploy the same function body as a Lambda — mirroring AgentCore's Lambda code-based evaluators (SPEC references) — so a code evaluator can also serve as an AgentCore custom code-based evaluator when SAES runs alongside the managed service.

---

## 7. Data Source — CloudWatch / OTEL Ingestion (framework-agnostic)

SAES consumes agent behavior from CloudWatch as populated by AgentCore Observability. **The only integration contract is the wire format: OpenTelemetry traces following the GenAI semantic conventions, landing in a CloudWatch log group.** Any agent that satisfies this contract is evaluable — whether it was built with Strands, LangGraph, a custom Python/TypeScript/Java/Go framework, or instrumented via OpenInference. SAES never imports or executes the agent's code and makes no assumption about its SDK; it reads spans off CloudWatch and normalizes them.

This is what makes SAES framework-agnostic in practice: **a third-party team's only job is to ship OTEL GenAI-convention traces to CloudWatch.** Once those spans exist, they can be evaluated by the same built-in evaluators, ground truth, and OpenAI-compatible judge as a first-party Strands agent — no code change on the agent side, no SDK adoption.

### 7.1 Prerequisites (the entire contract)

1. **Traces in OTEL GenAI format.** Spans carry the GenAI semantic-convention attributes (`gen_ai.*` — prompts, completions, tool calls, model params) and are grouped by a session identifier. OpenInference-instrumented spans are accepted via an attribute-mapping profile.
2. **Delivered to a CloudWatch log group** (for online eval; offline uses a local dump). Any log group the SAES role can read — the AgentCore default (`/aws/bedrock-agentcore/runtimes/<id>-DEFAULT`) or a custom one. *How* spans get there is the agent's choice (ADOT collector, OTLP→CloudWatch exporter, or direct `PutLogEvents`) — see §7.1b. ADOT is recommended but not required.
3. **CloudWatch Transaction Search enabled** (one-time account setup) so spans/traces are queryable for reconstruction.

No requirement that the agent run on AgentCore Runtime, use Strands, or be written in any particular language. If a fourth-party agent can emit these spans, SAES scores it.

### 7.1a Conformance check

`saes doctor --data-source <cfg>` samples recent spans and reports whether they satisfy the contract: presence of required `gen_ai.*` attributes, session/trace/span grouping keys, and tool-call attributes. It emits a per-field coverage report so a third-party team can see exactly which conventions their instrumentation is missing before running an evaluation, rather than getting silently empty results.

### 7.1b Getting spans into CloudWatch (ADOT is recommended, not required)

**SAES itself needs no telemetry SDK** — it neither imports OpenTelemetry nor depends on ADOT; it only *reads* spans that are already in CloudWatch. The delivery mechanism is entirely the agent's concern, and the only hard requirement is the wire format (§7.1). Verified delivery paths, in order of least effort:

| Agent host | How spans reach CloudWatch | ADOT? |
|---|---|---|
| **AgentCore Runtime** | AgentCore's container runs `aws-opentelemetry-distro` + `opentelemetry-instrument` and the managed runtime exports spans to `/aws/bedrock-agentcore/runtimes/<id>-DEFAULT`. This is the default scaffold — nothing extra to wire. | Yes (built in) |
| **Self-hosted / any framework** | Run an **ADOT collector** (or the OTLP → CloudWatch exporter) alongside the agent; point your OpenInference/LangChain/Strands instrumentation at it. Standard, production-grade. | Yes (recommended) |
| **Anything, minimal** | Emit OTEL-convention span records directly to a log group (e.g. `PutLogEvents` with the GenAI-convention JSON). No collector, no ADOT — verified in this project with a plain-Python, no-framework agent. | No |
| **Offline / CI** | No CloudWatch at all — hand SAES a local OTLP/JSONL dump via `dataSource.type: otlp_file`. | No |

So **ADOT is the recommended, lowest-friction path for production agents that don't already export to CloudWatch**, and it's built into the AgentCore Runtime path — but SAES never requires it. Any route that lands OTEL GenAI-convention spans in a readable log group (or a local dump) works. See §7.1c for the span-shape agents must emit so the native mappers reconstruct a full turn.

### 7.1c Emitting an evaluable turn (span shape)

The native session mappers reconstruct a turn from a **root/agent-invocation span** plus its child LLM/tool spans — the TRACE_LEVEL evaluators require the agent-invocation span (a bare LLM span alone is not enough). Emit a recognized root span in one of these conventions:

- **traceloop / LangChain-OTEL** — root span with `traceloop.span.kind = "workflow"` (+ `traceloop.entity.input/output`); child LLM spans with `llm.request.type = "chat"`.
- **OpenInference** — root `CHAIN` span; child `LLM`/`TOOL` spans with `llm.input_messages.*` / `llm.output_messages.*`.
- **CloudWatch body format** — what AgentCore Runtime emits natively (body-format events); handled automatically.

`saes doctor --data-source` confirms whether sessions reconstruct with the needed spans, so instrumentation gaps surface before a run rather than as empty scores.

### 7.2 Trace ingestion adapter

```
DataSourceConfig:
  type: cloudwatch                       # cloudwatch | otlp_file | langfuse | live
  cloudwatch:
    log_group_names: ["/aws/bedrock-agentcore/my-agent"]
    service_names: ["my-agent.DEFAULT"]
    region: us-east-1
    time_window: {relative: "PT1H"}      # or absolute start/end
    filter: "attributes.session.id = ..."  # optional
```

Sources (all delegate to native `strands-agents-evals` providers/mappers):
- **`cloudwatch`** — native `CloudWatchProvider` (reads AgentCore runtime log groups; discovers by `agent_name`). Framework-agnostic; works identically for Strands and non-Strands agents. (M2)
- **`otlp_file`** — local `.jsonl`/OTLP dump for offline dev and CI (no AWS needed); the SAES-owned thin reader. The format a third party can produce anywhere. (M1)
- **`langfuse`** / **`opensearch`** — native `LangfuseProvider` / `OpenSearchProvider`.
- **`live`** — native `@eval_task(TracedHandler())` in-process capture for Strands dev loops. (Strands-only convenience; third-party agents use `cloudwatch`/`otlp_file`.)

### 7.3 Normalization (native mappers, zero SAES mapping code)

Normalization is **entirely** `strands-agents-evals` **session mappers** — `CloudWatchSessionMapper`, `OpenInferenceSessionMapper`, `LangChainOtelSessionMapper`, `StrandsInMemorySessionMapper` — selected by `detect_otel_mapper()` (scope-name based). They map OTEL/GenAI (and OpenInference, LangChain-OTEL) spans into the native `Session` model that evaluators consume. SAES authors **no** mapping code and defines **no** canonical trace type of its own; the native `Session` *is* the model.

SAES owns only a thin `otlp_file` reader: read a local dump → group span dicts by `session.id` → hand each group to the auto-detected (or explicitly configured) native mapper → native `Session`. Trace fetching for production reuses the native `TraceProvider` implementations. This native mapper/provider layer is the seam that keeps everything independent of the trace source and the agent's framework — the integration contract for any agent is simply "emit OTEL GenAI-convention spans."

---

## 8. Evaluation Modes

### 8.1 On-demand (offline) evaluation

For development, CI/CD, regression testing, and ground-truth dataset runs.

- **Dataset runner** — iterate a ground-truth dataset, (optionally) execute the agent via a task function to produce fresh traces, then score. Mirrors the Strands `Experiment` / `run_evaluations` flow and AgentCore's `OnDemandEvaluationRunner`.
- **Existing-session runner** — score already-recorded sessions/traces/spans by ID from the data source. Mirrors AgentCore `EvaluationClient`.
- **Gate** — exit non-zero when aggregate scores fall below configured thresholds, for CI deployment gating.

```bash
saes run --config eval.yaml --dataset ground_truth.jsonl --gate helpfulness>=0.8,correctness>=0.9
```

Results: JSON report + optional CloudWatch emission + local HTML summary.

### 8.2 Online (continuous) evaluation

For production monitoring. A long-running worker (container/ECS/Lambda-on-schedule) polls the CloudWatch data source, detects **completed** sessions, samples them per `SamplingConfig`, scores them, and writes results back. This mirrors how managed AgentCore Evaluations works: it too reads spans from CloudWatch, groups them into whole-conversation sessions, and runs evaluators on a recurring schedule.

```yaml
mode: online
sampling:
  percentage: 5.0            # evaluate 5% of production sessions
  max_per_minute: 100        # rate cap to bound judge cost
  filters: ["service.name = 'my-agent.DEFAULT'"]
session:
  timeout_minutes: 30        # a session is "complete" this long after its last span
```

- Adds quality scores to existing telemetry without agent code changes or redeploys.
- Bounded by rate/percentage caps to control judge-model spend.
- Any drop/truncation due to caps is logged explicitly (never silent) so dashboards don't misrepresent coverage.

**Session-completion detection (span-quiescence timeout).** Sessions have no explicit end marker in OTEL traces, so the worker treats a session as complete only when **no new span has arrived for `session.timeout_minutes`** after its latest span. Sessions still receiving spans are skipped that cycle (they'd be scored mid-conversation and produce misleading results). This matches managed AgentCore's `SessionConfig` timeout, and the value should approximate the agent's typical session duration:
- Too short → in-progress conversations get scored prematurely.
- Too long → results lag real time.

The worker tracks each session's last-seen span timestamp across polling cycles; only sessions whose quiescence exceeds the timeout (and haven't already been scored) enter the sampling/scoring pipeline. Each session is scored at most once.

---

## 9. Automated Test Generation & Detectors (optional, from Strands Evals)

- **Experiment generation** — generate test suites from a context description (tool set + domain), using the configured judge/generator model, to bootstrap datasets.
- **Simulators** — multi-turn conversation simulation with goal-oriented synthetic users and LLM-powered tool simulation, for stress-testing multi-turn behavior.
- **Detectors** — automatic failure detection, root-cause analysis, and fix recommendations over traces.

These reuse `strands-agents-evals` capabilities and honor the same OpenAI-compatible judge/generator configuration.

---

## 10. Results, Metrics & Dashboards

### 10.1 Results sink

```yaml
resultsSink:
  cloudwatch:
    log_group: "/aws/saes/evaluations"     # structured JSON results
    metrics_namespace: "SAES/Evaluations"  # EMF metrics
    dimensions: [agentId, evaluatorId, env]
  local:
    json_path: "./out/results.json"
    html_report: "./out/report.html"
```

- **CloudWatch metrics (EMF)** — per-evaluator score statistics (avg, p50, pass-rate) emitted as custom metrics so they appear alongside operational metrics (latency, token usage, error rate) in AgentCore Observability. Enables CloudWatch alarms on quality regressions (e.g. alarm when `Helpfulness` avg < 0.75).
- **CloudWatch Logs** — full `EvaluationResult` records in JSON for drill-down, including judge reasoning per trace.
- **Local** — JSON + self-contained HTML report for CI artifacts and offline review.

### 10.2 Dashboard integration

Because results land in CloudWatch, they render in the CloudWatch generative-AI observability views: agent-level aggregate trends and drill-down to individual sessions/traces showing scores and judge reasoning. SAES ships an optional CloudWatch dashboard template (and CDK construct) provisioning score trend widgets and regression alarms.

---

## 11. Architecture

```
                        ┌────────────────────────────────────────────┐
                        │  Strands Agent (instrumented, OTEL/GenAI)    │
                        └───────────────────┬──────────────────────────┘
                                            │ spans/traces
                                            ▼
                        ┌────────────────────────────────────────────┐
                        │  AgentCore Observability  →  CloudWatch      │
                        │  (Transaction Search, Logs, gen-AI dashboard)│
                        └───────────────────┬──────────────────────────┘
                                            │ read (adapter)
        ┌───────────────────────────────────┼───────────────────────────────────┐
        │                            SAES Core (OSS)                              │
        │  ┌────────────┐  ┌──────────────┐  ┌────────────────────────────────┐  │
        │  │ Data Source│→ │ native mapper│→ │ strands-agents-evals (native)  │  │
        │  │ = native   │  │ detect_otel_ │  │  • 13 built-in evaluators      │  │
        │  │ providers  │  │ mapper →     │  │  • trajectory scorers          │  │
        │  │ + thin     │  │ native       │  │  • Experiment/Case/Report      │  │
        │  │ otlp_file  │  │ Session      │  │  • generation / detectors      │  │
        │  │ ANY OTEL → │  └──────────────┘  └───────────────┬────────────────┘  │
        │  └────────────┘   SAES resolves ids/custom → judge │ judge calls        │
        │  ┌──────────────────────────────────────────────┐ │                    │
        │  │ Judge Model Layer (build_model → strands)     │◄┘                    │
        │  │  openai_compatible | bedrock                  │                      │
        │  └──────────────────────────────────────────────┘                      │
        │  ┌────────────┐   ┌──────────────┐   ┌───────────────────────────────┐ │
        │  │ Config /   │   │ On-demand    │   │ Online worker (sampling loop) │ │
        │  │ CLI (saes) │   │ runner + gate│   │ container/ECS/Lambda          │ │
        │  └────────────┘   └──────────────┘   └───────────────────────────────┘ │
        │                          │ results                                      │
        └──────────────────────────┼──────────────────────────────────────────────┘
                                   ▼
                 CloudWatch (EMF metrics + JSON logs)  +  local JSON/HTML report
```

### 11.1 Packaging

- `saes-core` — Python 3.10+ library: judge layer, config, evaluator resolution (ids→native classes + custom LLM/code glue), thin `otlp_file` reader, CloudWatch results sink. **All evaluators, prompt templates, session mappers, trace providers, the `Experiment`/`Case`/`Report` engine, generation, and detectors are provided by `strands-agents-evals` — SAES reimplements none of them.** Depends on `strands-agents`, `strands-agents-evals`, `strands-agents[openai]`, `boto3`. SAES-owned code is intentionally thin: judge config/resolution, evaluator registry + custom-evaluator glue, source factory, online worker, results sink.
- `saes-cli` — `saes init | run | serve | dataset | report`.
- `saes-worker` — online evaluation daemon + Lambda handler.
- `saes-cdk` — optional CDK constructs (dashboard, alarms, Lambda code-evaluator wiring, IAM roles).

---

## 12. Configuration Reference (consolidated example)

```yaml
name: my-agent-quality
mode: on_demand

dataSource:
  type: cloudwatch
  cloudwatch:
    log_group_names: ["/aws/bedrock-agentcore/my-agent"]
    service_names: ["my-agent.DEFAULT"]
    region: us-east-1
    time_window: {relative: "PT6H"}

judge:
  provider: openai_compatible
  model: "gpt-4.1"
  base_url: "https://llm-gateway.internal/v1"
  api_key_env: "SAES_JUDGE_API_KEY"
  params: {temperature: 0.0, max_tokens: 1024}
  structured_output: json_schema

evaluators:
  - Builtin.Helpfulness
  - Builtin.Correctness           # uses expectedResponse when present
  - Builtin.ToolSelectionAccuracy
  - Builtin.GoalSuccessRate        # uses assertions
  - id: hipaa_compliance           # custom LLM evaluator
    level: trace
    type: llm
    scale: binary
    instructions: "..."

groundTruth:
  path: ./ground_truth.jsonl

resultsSink:
  cloudwatch:
    log_group: "/aws/saes/evaluations"
    metrics_namespace: "SAES/Evaluations"
    dimensions: [agentId, evaluatorId]
  local:
    html_report: ./out/report.html

gate:
  - "Builtin.Helpfulness.avg >= 0.8"
  - "Builtin.Correctness.avg >= 0.9"
```

---

## 13. Security & Operations

- **Secrets** — judge API keys referenced by env var / Secrets Manager only; never inline in config or logs. The judge base_url and key are redacted from result provenance.
- **IAM (least privilege)** — read: CloudWatch Logs/Transaction Search on the agent log groups; write: `PutLogEvents` + `PutMetricData` on the SAES results namespace; invoke: Lambda code evaluators. Online worker uses a dedicated execution role.
- **Data residency** — with a self-hosted OpenAI-compatible judge, trace content never leaves your network. With external judge endpoints (OpenAI, etc.), trace content **is** sent to that provider — flagged prominently in docs and by a config-time warning; an allowlist of approved judge hosts can be enforced.
- **Cost controls** — online sampling percentage + per-minute caps; per-evaluator judge overrides (cheap models for high-volume evaluators); deterministic code evaluators preferred where language understanding isn't needed.
- **Determinism** — judge `temperature: 0.0` recommended; results stamp `judgeModel` and prompt-template version for reproducibility.
- **Failure handling** — judge parse/timeout failures retried up to `max_retries`, then marked `errored` with the raw response retained; errored results are excluded from aggregates but reported in run summary counts.

---

## 14. Milestones

| Phase | Status | Deliverable |
|---|---|---|
| **M1 — Core** | ✅ done | Config schema, judge layer (openai_compatible + bedrock), `otlp_file` source over native mappers, full built-in + custom LLM/code evaluators, on-demand runner + gate, local JSON/HTML report. |
| **M2 — Observability** | ✅ done | CloudWatch trace source + session discovery, EMF/log results sink, all 13 built-in evaluators + trajectory scorers verified, ground-truth dataset support, `saes doctor`. |
| **M3 — Online** | ✅ done | Sampling worker (`saes serve`, span-quiescence completion), Lambda code evaluators, CloudWatch dashboard/alarms CDK. |
| **M4 — Generation** | planned | Experiment generation, simulators, detectors; Langfuse adapter; promote-trace-to-dataset flow. |

(Statuses reflect implementation; see the per-milestone PLAN docs and DOCUMENTATION.md §10. All native `strands-agents-evals` capabilities used by M1–M3 are verified; M4 items are additional native capabilities not yet surfaced through SAES.)

---

## 15. Resolved Decisions

- **D1 — Reuse the engine.** SAES builds on `strands-agents-evals` and does **not** reimplement the evaluation engine, evaluators, scorers, generation, or detectors. SAES-owned code is limited to: the CloudWatch/OTEL ingestion adapter + attribute-mapping profiles, the OpenAI-compatible judge layer, the online worker, and the CloudWatch results sink. Minimal net-new surface. *(Trade-off accepted: SAES tracks the upstream package's API and roadmap.)*
- **D2 — Framework-agnostic input contract.** The sole integration requirement for any evaluated agent is: emit OTEL GenAI-convention traces to a CloudWatch log group. No Strands SDK, AgentCore Runtime, or language requirement on the agent. Third-party agents are first-class via the `cloudwatch`/`otlp_file` adapters and `saes doctor` conformance check.

## 15b. Open Questions

1. **~~Judge parity across endpoints~~ — RESOLVED (Decision 1):** the judge must support tool-calling / structured output (the native evaluators require it). SAES states this requirement explicitly (§1, §3.2) and enforces it with a preflight capability probe (§3.5, `saes doctor --judge` / `saes init`) that fails clearly rather than mid-run. Text-only endpoints are unsupported by design, not silently broken.
2. **Baseline comparison across judges** — beyond a warning, should SAES offer a calibration harness (score a fixed golden set with the new judge to derive an offset)?
3. **Session reconstruction fidelity** — edge cases where CloudWatch Transaction Search sampling drops spans; the completeness check flags partial sessions rather than scoring them as-is (open: default policy — skip vs. score-with-warning).
4. **Native dashboard alignment (deferred, not blocking)** — should SAES *optionally* also emit results in AgentCore Evaluations' native CloudWatch schema so both render in the same managed dashboard widgets? Default remains SAES's own `SAES/Evaluations` namespace (self-owned, not coupled to an undocumented managed schema); the native-schema emitter is a possible later add-on, not the primary path.

---

## 16. References

- [Observe your agent applications on Amazon Bedrock AgentCore Observability](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html)
- [Build reliable AI agents with Amazon Bedrock AgentCore Evaluations (AWS ML Blog)](https://aws.amazon.com/blogs/machine-learning/build-reliable-ai-agents-with-amazon-bedrock-agentcore-evaluations/)
- [AgentCore Ground truth evaluations](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/ground-truth-evaluations.html)
- [AgentCore Built-in evaluators](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/built-in-evaluators-overview.html)
- [AgentCore Create online evaluation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/create-online-evaluations.html)
- [AgentCore Custom code-based evaluator](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-based-evaluators.html)
- [Strands Evaluation quickstart](https://strandsagents.com/docs/user-guide/evals-sdk/quickstart/index.md)
- [Strands OutputEvaluator](https://strandsagents.com/docs/user-guide/evals-sdk/evaluators/output_evaluator/index.md)
- [Strands OpenAI model provider](https://strandsagents.com/docs/user-guide/concepts/model-providers/openai/index.md)
- [Strands observability & traces](https://strandsagents.com/docs/user-guide/observability-evaluation/observability/index.md)
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
