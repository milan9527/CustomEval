---
name: otel-eval-contract
description: The OTEL trace contract every AI agent must follow — regardless of framework (Strands, LangGraph, CrewAI, Claude Agent SDK, bare boto3, custom) — so its traces can be fully evaluated by SAES / strands-agents-evals (all 15 built-in evaluators run). Trigger when writing agent code, adding instrumentation/telemetry to an agent, deploying to AgentCore Runtime, debugging evaluators that produce no scores / SessionNotFoundError, or any mention of "OTEL format", "trace contract", "evaluable agent".
---

# The OTEL evaluation contract — every agent must follow it

**Purpose**: no matter which framework generates an agent, the OTEL spans it
emits must carry the fields below so the evaluation side (SAES, built on the
`strands-agents-evals` mappers plus SAES's supplements) can reconstruct an
evaluable session from the trace. **Adaptation lives in evaluation-side
ingestion, not in the agent — the agent only has to guarantee the raw data is
in the spans.**

Authoritative sources: `DOCUMENTATION.md` §7.4 and
`src/saes/ingest/conformance.py` (`_CHECKS`).
If this file ever disagrees with those, the code wins.

## 1. The universal contract (every framework, every agent)

When generating any agent, verify the spans it emits satisfy all 5:

| #   | Requirement                                           | Accepted attribute keys (any one)                                                                                                                    |
| --- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **session id** — how spans group into a conversation  | `session.id`, `gen_ai.session.id`, `session_id`                                                                                                      |
| 2   | **prompt / input text**                               | `gen_ai.prompt` (or indexed `gen_ai.prompt.N.content`), `gen_ai.input.messages`, `input.value`, `llm.input_messages.*`, `traceloop.entity.input`     |
| 3   | **completion / output text**                          | `gen_ai.completion` (or `gen_ai.completion.N.content`), `gen_ai.output.messages`, `output.value`, `llm.output_messages.*`, `traceloop.entity.output` |
| 4   | **`scope.name`** — selects the native mapper (see §2) | the span's instrumentation scope                                                                                                                     |
| 5   | **`traceId` + `spanId`**, and **one trace per turn**  | standard OTEL; multi-turn sessions are split into turns by trace and ordered by span time                                                            |

- Minimum to reconstruct anything: **session id + (prompt OR completion)**.
- On AgentCore Runtime the session id comes from
  `agentcore invoke --session-id <id>`; a multi-turn session = reusing the same
  session id across invocations.
- Tool name (needed for tool-level evaluation): any of `gen_ai.tool.name`,
  `tool.name`, `tool_call.function.name`.

## 2. `scope.name` selects the mapper — an unknown scope is NOT broken

The native mappers recognize only three scopes:

| `scope.name`                                   | Result                                             |
| ---------------------------------------------- | -------------------------------------------------- |
| `strands.telemetry.tracer`                     | Strands mapper, full coverage (agent + tool spans) |
| `opentelemetry.instrumentation.langchain`      | LangChain-OTEL mapper                              |
| `openinference.instrumentation.langchain`      | OpenInference mapper                               |
| anything else (`…crewai`, `botocore…`, custom) | native read may raise — **expected, not a bug**    |

When the scope is not one of the three, SAES's supplements recover the
trajectory and turns from the raw Bedrock Converse spans (botocore
`toolUse`/`toolResult`, roled `body.message`). **Just make sure Bedrock calls
go through the instrumented client** (the default on AgentCore Runtime). Do
not write your own mapper, and do not change the agent to fake a known scope.

## 3. What unlocks each evaluator level

| Level                            | Example evaluators                           | The agent must emit                                                                                                                                                        |
| -------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trace-level                      | Helpfulness, Correctness, Coherence, …       | the user prompt + the **final answer** (contract items 2+3)                                                                                                                |
| Tool-level                       | ToolSelectionAccuracy, ToolParameterAccuracy | the tool call's **name + arguments + result**. Non-Strands agents only need the Converse `toolUse`/`toolResult` blocks in the spans; arguments must be the real tool input |
| Trajectory match (deterministic) | TrajectoryAnyOrderMatch                      | ordered tool-call names (from the same `toolUse` blocks) + an `expectedTrajectory` in ground truth                                                                         |

## 4. Per-framework instructions (verified on real deployments)

- **Strands** — nothing to do. Native OTEL emits `AgentInvocationSpan` +
  `ToolExecutionSpan` + `InferenceSpan`; all 15 evaluators run, multi-turn
  included.
- **LangGraph** — enable OpenInference: `LangChainInstrumentor().instrument()`
  (or the OTEL LangChain instrumentor) so the scope is one of the two LangChain
  values. Tool calls flow out through Bedrock Converse spans; SAES recovers
  them. With `ChatBedrockConverse` on an inference-profile ARN, set
  `provider="anthropic"`.
- **CrewAI** — its scope is `openinference.instrumentation.crewai`, so the
  native read raises (expected); SAES recovers from the Converse spans. Known
  gap: the per-turn user prompt is not always in a recoverable shape, so
  ResponseRelevance can miss.
- **No framework (bare boto3)** — add no instrumentation: AgentCore's botocore
  Bedrock instrumentation already captures the Converse request/response (the
  final answer lands in `body.message`). Just keep Bedrock calls on the
  default (instrumented) client.
- **Claude Agent SDK** — the custom path is REQUIRED. The SDK drives Bedrock
  through a bundled CLI **subprocess**, so AgentCore's botocore instrumentation
  never sees the model calls — nothing is captured for free. Verified working
  recipe (`examples/agents/claudesdk_tools/agent.py`,
  deployed + evaluated end-to-end):
  - one root span per invocation with `session.id` (from AgentCore's
    `RequestContext.session_id`), `gen_ai.prompt`, `gen_ai.completion`;
  - additionally emit one OTEL **event** per turn whose body is roled
    `{"input": {"messages": [{role: "user", content}]}, "output": {"messages":
[{role: "assistant", content}]}}` — the authoritative shape SAES's
    role-aware recovery reads;
  - per tool call, emit Converse-shaped `toolUse`/`toolResult` event bodies
    (real arguments + real result) — the tool supplement recovers the
    trajectory and synthesizes tool spans from these;
  - SDK tool callbacks run in the SDK's own asyncio tasks, so parent their
    spans/events on the root span via an explicit `SpanContext` — contextvar
    propagation does NOT cross the CLI transport;
  - expose tools via the in-process MCP server (`create_sdk_mcp_server`) and
    disable the built-in tools (Bash/Read/Write/…) with `disallowed_tools`;
  - Dockerfile: the bundled CLI needs `ca-certificates` + `ripgrep` installed
    and a writable `HOME`; set `CLAUDE_CODE_USE_BEDROCK=1` in `options.env`.
- **Custom framework / not on AgentCore** — satisfy the contract by hand with
  the OTEL SDK; minimal example:

```python
from opentelemetry import trace
tracer = trace.get_tracer("my.custom.agent")   # scope.name — any value; supplements recover it

# one trace per turn
with tracer.start_as_current_span("invoke_agent") as span:
    span.set_attribute("session.id", session_id)          # contract item 1
    span.set_attribute("gen_ai.prompt", user_prompt)      # contract item 2
    # ... run the agent; emit one child span per tool call:
    #   gen_ai.tool.name / arguments / result (or leave the Bedrock Converse
    #   toolUse/toolResult blocks on the instrumented call span)
    span.set_attribute("gen_ai.completion", final_answer) # contract item 3
```

## 5. Mandatory verification (do not trust scores without it)

After generating or changing an agent, and before trusting any evaluation
scores, run:

```bash
saes doctor --data-source <dump.jsonl>
```

It prints per-field contract coverage: a `✓` on session id + prompt +
completion means sessions reconstruct; a `✗` names exactly which attribute the
instrumentation is missing — **fix it at the source (the agent's
instrumentation), never by patching the evaluation side.**

Reading doctor output on supplement-path agents (Claude Agent SDK, CrewAI,
no-framework): prompt/completion may show `✗` in the _field-coverage_ rows when
the content lives in event **bodies** rather than span attributes — that is
fine as long as the final "sessions reconstructed" line succeeds (exit 0). The
field rows check attributes; the supplement reads bodies. To confirm the deep
reconstruction, run `extract_session_tool_calls` on the dump and check each
turn pairs its own prompt + answer + tools.

Before deploying, smoke-test locally: run one invocation under an in-memory
OTEL exporter (`InMemorySpanExporter` + `InMemoryLogExporter`), write the
records as JSONL, and run `saes doctor` on that — catches contract violations
without a ~10-minute deploy cycle. Working example:
`examples/agents/claudesdk_tools/smoke_local.py`.

AgentCore caveat: after an invoke, CloudWatch trace delivery + Logs Insights
indexing lag by ~90–100 seconds. If a session doesn't show up, wait before
concluding the instrumentation is broken. The log group is always
`/aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT`.

## 6. Checklist for any agent-generating agent

Before delivering any generated agent code, self-check:

- [ ] Every span carries a session id (one of the three aliases)
- [ ] A prompt attribute and a completion (final answer) attribute are emitted
- [ ] One trace per turn; multi-turn reuses the same session id
- [ ] Tool call name/arguments/result are in the spans (native tool spans or Converse `toolUse`/`toolResult` both work)
- [ ] Bedrock calls go through the instrumented client (no bare client that bypasses auto-instrumentation) — or, if the framework calls Bedrock out-of-process (Claude Agent SDK's CLI subprocess), the contract is satisfied by hand per §4
- [ ] No mapping/transformation code was written for evaluation — that belongs to the evaluation side
- [ ] A real dump was verified with `saes doctor` before delivery
