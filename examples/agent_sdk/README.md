# Claude Agent SDK → OTLP → CloudWatch → `saes eval` (real end-to-end)

The fifth framework in the SAES matrix, and the one that proves the OTLP→CloudWatch
emit path end to end: a **Claude Agent SDK** agent (`claude-agent-sdk`, Claude Code
as a library) with its built-in OpenTelemetry, exporting over OTLP to an ADOT
collector that forwards to CloudWatch, then evaluated on-demand with `saes eval` —
no YAML, no ground truth, just the log group.

This is the sequel to the raw-`anthropic`-SDK finding (that SDK bypasses botocore
and emits nothing — see `../../` note / `ANTHROPIC_SDK_FINDING.md` in the run
workspace). The **Agent** SDK is a different story: it has rich, session-tagged
built-in telemetry, and SAES now parses its real CloudWatch record shape.

## The full pipeline — from agent run to evaluation result

There are **two opposite directions** here, and keeping them straight is the whole
mental model:

```
┌─ EMIT (runtime) ────────────────────────┐   ┌─ EVALUATE (offline, SAES side) ─────────┐
│                                          │   │                                          │
│  Claude Agent SDK        (run_agent.py)  │   │  saes eval /aws/saes/agentsdk-results    │
│    │  built-in OpenTelemetry             │   │    │                                     │
│    │  emits claude_code.* events         │   │    │ 1. discover session ids in the group │
│    ▼  OTLP/http → :4318                  │   │    │ 2. fetch raw span records            │
│  ADOT collector (Docker, collector.yaml) │   │    │ 3. reconstruct the turn(s)           │
│    │  awscloudwatchlogs exporter         │   │    │    (prompt · final answer · tools)   │
│    ▼                                     │   │    │ 4. score each turn with an LLM judge │
│  CloudWatch Logs ────────────────────────────▶ 5. write report to LOCAL disk           │
│  /aws/saes/agentsdk-results              │   │       out/agentsdk_report.{html,json}    │
│  (raw telemetry ONLY — the INPUT)        │   │       (the OUTPUT — NOT in CloudWatch)   │
└──────────────────────────────────────────┘   └──────────────────────────────────────────┘
```

Step by step:

1. **Agent runs** — `run_agent.py` runs a `claude_agent_sdk.query(...)`. With
   `CLAUDE_CODE_ENABLE_TELEMETRY=1`, the SDK's built-in OpenTelemetry emits each step
   (user prompt, API request/response bodies, tool decision, tool result, final
   answer) as `claude_code.*` events over **OTLP** (it does not write to CloudWatch
   directly, and its docs warn against the `console` exporter).
2. **Collector forwards** — the ADOT collector (otel-contrib in Docker) receives OTLP
   and its `awscloudwatchlogs` exporter writes **each raw event** as one log record
   into `/aws/saes/agentsdk-results`.
3. **CloudWatch holds the raw telemetry** — ~56 `claude_code.*` JSON records. This is
   the **input** to evaluation, not a result.
4. **SAES evaluates** — `saes eval` reads that group back: discovers the sessions,
   fetches the raw records, reconstructs the clean turn, and sends each turn to the
   LLM judge for the 12 evaluators.
5. **Result lands on local disk** — the terminal table plus whatever `--html` / `--json`
   files you asked for (here `out/agentsdk_report.html` + `.json`). **`saes eval` does
   not write results back to CloudWatch.**

### Why you don't see the HTML report inside `/aws/saes/agentsdk-results`

Because that log group is the agent's **input**, not SAES's **output** — they flow in
opposite directions:

| | Content | Written by | Direction |
|---|---|---|---|
| `/aws/saes/agentsdk-results` | agent's raw OTEL telemetry events | ADOT collector (agent side) | **into** CloudWatch |
| the HTML/JSON report | evaluation scores + reasoning | `saes eval` (local process) | read out → **local disk** |

`saes eval` is an offline **reader**: it pulls raw records *out* of CloudWatch, scores
them locally, and writes the report to `out/`. Analogy: the log group is the security
camera's **tape**; the HTML report is the **inspection write-up** you produce after
watching it — the write-up never appears back on the tape.

**If you want the scores in CloudWatch/AWS**, use online mode — it writes results to a
**separate results log group**:

```bash
saes serve /aws/saes/agentsdk-results --results-log-group /aws/saes/agentsdk-scores
```

Results and raw telemetry stay in two different groups; scores never get mixed into the
raw spans. (On-demand `saes eval` is the local-report path; to view a report in AWS,
upload the HTML to S3 static hosting.)

## Files

| File | What it is |
|---|---|
| `run_agent.py` | the agent — `claude_agent_sdk.query(...)` with the OTEL env from the [SDK observability docs](https://code.claude.com/docs/en/agent-sdk/observability): `CLAUDE_CODE_ENABLE_TELEMETRY=1`, OTLP exporters → `localhost:4318`, `OTEL_LOG_RAW_API_BODIES=1`, running on Bedrock |
| `collector.yaml` | otel-contrib ADOT collector: OTLP receiver → `awscloudwatchlogs` exporter (log group `/aws/saes/agentsdk-results`) + `debug` |
| `requirements.txt` | `claude-agent-sdk` |
| `AGENT_SDK_EVAL_OUTPUT.txt` | verbatim `saes eval` terminal output |
| `AGENT_SDK_EVAL_OUTPUT.json` | the same run's machine-readable result (per-session `rows` + aggregates) |
| `AGENT_SDK_REPORT.html` | the same run rendered as a browsable report (aggregates + per-result detail with the reconstructed conversation, tool calls, and judge reasoning) — open in a browser |

## Reproduce

```bash
# 1. run the collector (needs AWS creds with logs:PutLogEvents on the log group)
docker run -d --name saes-otelcol -p 4318:4318 -p 4317:4317 \
  -v "$PWD/collector.yaml:/etc/otelcol-contrib/config.yaml" \
  -e AWS_REGION=us-east-1 \
  otel/opentelemetry-collector-contrib:latest

# 2. run the agent (emits claude_code.* OTEL → collector → CloudWatch)
pip install -r requirements.txt
python run_agent.py

# 3. evaluate — just the log group, no config
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
saes eval /aws/saes/agentsdk-results --lookback-days 1
```

## Result (3 real sessions, 12 reference-free evaluators)

```
eval-agentsdk-results  (judge: openai.gpt-oss-20b-1:0)  [3 session(s)]
  Helpfulness  0.889 · Coherence/Conciseness/Faithfulness/InstructionFollowing 1.0
  ResponseRelevance/ContextRelevance/Harmfulness/Stereotyping 1.0
  Refusal 0.0  (correct — the agent answered, it did not refuse)
  ToolSelectionAccuracy 1.0 (n=2)   ← correctly picked Bash
  ToolParameterAccuracy 0.5 (n=2)
```

Every evaluator produced real scores (`n=3`, or `n=2` for the tool evaluators since
only 2 of the 3 sessions used a tool). The reconstructed turn is faithful:

- prompt: `"Run the bash command echo hello-from-agent and tell me its output."`
- final answer: `"The output is: hello-from-agent"` (the real final turn, **not** the
  intermediate `"I'll run that command for you."` nor the CLI's internal
  session-title JSON)
- trajectory `["Bash"]`, args `{command: "echo hello-from-agent"}`

## Honest notes

- **`ToolParameterAccuracy` is non-deterministic here.** Across identical re-runs the
  gpt-oss-20b judge has scored this borderline case 0.0 / 0.5 / 1.0. The `command`
  argument round-trips faithfully every time; the wobble is the judge reacting to the
  Bash tool's real `description` argument, which isn't in the tool schema SAES
  synthesizes from *observed* calls (SAES infers `available_tools` from the calls it
  sees, without each tool's full parameter spec). This is a judge/schema-fidelity
  nuance, not an ingestion defect — pin a stronger, more deterministic judge for a
  stable number.
- **Tool *result* text is not recoverable.** The SDK's `claude_code.tool_result` log
  event carries only metadata (`tool_use_id`, `tool_name`, `success`, `duration_ms`,
  sizes) — not the result payload. Trajectory, args, prompt, and final answer are all
  recovered, which is what the trajectory / turn / tool-selection evaluators consume.

## What SAES had to learn (the real record shape)

The live CloudWatch records differ from the documented per-scope `claude_code.*`
shape; the ingestion fixes are in `src/saes/ingest/`:

- one scope for everything (`com.anthropic.claude_code.events`); event kind is in
  `attributes["event.name"]` → SAES reads the dedicated `user_prompt` /
  `assistant_response` events (text lives directly in attributes);
- turns are keyed by `attributes["prompt.id"]`, **not** a `trace_id` (the log events
  carry none) → `_turn_id_of` prefers `prompt.id`, AgentCore still falls back to
  `trace_id`;
- the CLI's internal session-title call (`query_source="generate_session_title"`) is
  excluded so the scored answer is the real turn;
- records have no `timeUnixNano` → ordered by `event.sequence`;
- `fetch_session_records` now fetches by `session.id` directly (then bridges via
  `trace_id` for session-id-less tool spans) — the earlier trace_id-only fetch
  returned nothing for these records.

Regression test against these real records:
`tests/test_tool_supplement.py::test_recovers_turn_from_real_claude_agent_sdk_records`
(fixture `tests/fixtures/claude_agent_sdk_cloudwatch_records.json`).
