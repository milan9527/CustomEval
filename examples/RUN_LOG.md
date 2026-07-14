# SAES — Full Run Log (fresh independent environment)

A complete end-to-end run of SAES in a **brand-new, isolated venv**, from
installing the package to a scored evaluation report. Every command and its real
output is recorded below. Environment and artifacts are **preserved** under:

- venv:  `/home/ec2-user/saes_run_venv`
- work:  `/home/ec2-user/saes_run`  (this file, the agent, traces, config, `out/`)

Judge = Amazon Bedrock's OpenAI-compatible endpoint (`openai.gpt-oss-20b-1:0`).
Agent = a plain-Python, **no-framework** script calling Bedrock directly and
emitting OTEL traceloop-convention spans. Nothing is stubbed.

---

## Step 0 — Create an isolated venv and install SAES

```bash
python3.12 -m venv /home/ec2-user/saes_run_venv
source /home/ec2-user/saes_run_venv/bin/activate      # ← activate FIRST
pip install -e '/home/ec2-user/project/eval[dev]' openai aws-bedrock-token-generator
```

Clean-env check before install → `ModuleNotFoundError: No module named 'strands'`
(confirms the venv is empty). Install exit code: `0`.

Verify:
```
saes 0.1.0 | strands-agents 1.47.0 | strands-agents-evals 1.0.2 | openai 2.45.0
saes commands: run | doctor | init | serve
```

> `aws-bedrock-token-generator` is only needed to mint a Bedrock bearer token for
> the OpenAI-compatible judge; it is not a SAES dependency.

## Step 1+2 — Build a no-framework agent, run it, emit OTEL traces

`my_agent.py` is plain Python: it calls `bedrock-runtime` directly (no Strands /
LangChain / CrewAI) and writes OTEL traceloop-convention spans (a `workflow` root
+ a child `chat` LLM span per turn) to `traces.jsonl`.

```bash
python my_agent.py
```
Output:
```
  [s-geo] Q: What is the capital of France? One sentence.
         A: The capital of France is Paris.
  [s-math] Q: What is 25 times 4?
         A: 25 times 4 is 100.
wrote 4 spans / 2 sessions -> traces.jsonl
```

## Step 3 — Verify the traces (`saes doctor --data-source`)

```bash
saes doctor --data-source traces.jsonl
```
```
spans read: 4
field coverage:
  ✓ session id                       4/4
  ✓ scope name (mapper selection)    4/4
  ✓ prompt / input                   4/4   (matches traceloop.entity.input + indexed gen_ai.*)
  ✓ completion / output              4/4
  ✗ tool name                        0/4   (correct — this agent uses no tools)
  ✓ trace id                         4/4
  ✓ span id                          4/4

OK — 2 session(s) reconstructed
  s-geo: 1 trace(s)
  s-math: 1 trace(s)
```

## Step 4 — Verify the judge (`saes doctor --judge`)

```bash
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
saes doctor --judge eval.yaml
```
```
probing judge: openai_compatible / openai.gpt-oss-20b-1:0
  ✓ structured output confirmed via tool calling
```

## Step 5 — Config + ground truth

`eval.yaml` (Bedrock OpenAI-compatible judge, 2 evaluators, 2 gate rules) and
`gt.jsonl` (expected responses for both sessions). See files in this directory.

## Step 6 — Evaluate (`saes run`)

```bash
saes run -c eval.yaml --json out/results.json --html out/report.html
```
```
full-run-demo  (judge: openai.gpt-oss-20b-1:0)
  Builtin.Helpfulness              avg=0.833  pass=100%  n=2
  Builtin.Correctness              avg=1.000  pass=100%  n=2

JSON  → out/results.json
HTML  → out/report.html

GATE PASSED
  ✓ Builtin.Correctness.avg >= 0.9  (actual=1.000)
  ✓ Builtin.Helpfulness.avg >= 0.7  (actual=0.833)
```
Exit code: `0` (non-zero would mean a gate failed).

## Result

- `out/results.json` — overall 0.9165, per-result rows with **real judge
  reasoning** from gpt-oss-20b (e.g. *"Multiplying 25 by 4 yields 100. The
  assistant correctly states…"*).
- `out/report.html` — self-contained report (~6 KB) with per-result reasoning
  drill-down.

Full chain, all real: no-framework agent → real Bedrock generation → OTEL traces
→ native session mapper → native evaluators → Bedrock OpenAI-compatible judge →
aggregation → gate → JSON/HTML.

## Re-run in this preserved environment

```bash
source /home/ec2-user/saes_run_venv/bin/activate
cd /home/ec2-user/saes_run
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
python my_agent.py                              # regenerate traces (optional)
saes doctor --data-source traces.jsonl
saes doctor --judge eval.yaml
saes run -c eval.yaml --json out/results.json --html out/report.html
```

---

# Part 2 — ONLINE path (agent → CloudWatch → evaluate)

The Part 1 run above is the **offline** path (agent writes a local trace dump).
This part is the **online / production** path you asked for: the agent runs on
**AgentCore Runtime, which auto-exports OTEL traces to CloudWatch**, and SAES
discovers + reads them from CloudWatch, evaluates, and writes results back to
CloudWatch. Nothing hand-fed — real deployed agent, real CloudWatch.

## Deploy a real AgentCore agent (auto OTEL → CloudWatch)

`online_agent/agent.py` is a Strands agent behind `BedrockAgentCoreApp`.

```bash
agentcore configure -e agent.py -n saesonline -rf requirements.txt --create
# (enable ecr_auto_create in .bedrock_agentcore.yaml)
agentcore deploy -env BEDROCK_MODEL_ID=...
```
→ runtime `saesonline-OcwwkJGnaM-DEFAULT`; traces land in
`/aws/bedrock-agentcore/runtimes/saesonline-OcwwkJGnaM-DEFAULT` automatically
(AgentCore ships ADOT; no manual telemetry wiring).

## Invoke it (produces real traces in CloudWatch)

```bash
agentcore invoke '{"prompt": "What is the capital of France? One sentence."}'
agentcore invoke '{"prompt": "What is 25 times 4?"}'
agentcore invoke '{"prompt": "Name the largest ocean."}'
```
```
{"result": "The capital of France is Paris.\n"}
{"result": "25 times 4 equals 100.\n"}
{"result": "The largest ocean is the Pacific Ocean.\n"}
```

## SAES discovers the session FROM CloudWatch (Logs Insights)

```
[('5b17ea26-fe6a-4eac-ab3e-51b5d269c749', 1783777445385)]   # (session_id, last_span_ms)
```
(~90s wait for trace delivery + Logs Insights indexing.)

## Online evaluation: saes serve

`online.yaml` uses `dataSource.type: cloudwatch` + a `resultsSink.cloudwatch`.

```bash
saes serve -c online.yaml --once
```
```
serving online eval for 'saes-online-demo' (timeout=1.0m, sampling=100.0%)
  scored 1/1 session(s) this cycle
cycle: ready=1 scored=1 deferred=0 errored=0     # exit 0
```

serve = discover sessions in CloudWatch → detect completed (span-quiescence
timeout) → read spans back via native CloudWatchProvider → evaluate with the
Bedrock OpenAI-compatible judge → write results to CloudWatch.

## Results written BACK to CloudWatch (`/aws/saes/online-demo-results`)

```
EMF metric events: 2
  ns=SAES/Evaluations | Builtin.Helpfulness | Score=0.833 PassRate=1.0
  ns=SAES/Evaluations | Builtin.Correctness | Score=1.0    PassRate=1.0
result records (with judge reasoning): 2
  Builtin.Helpfulness [5b17ea26]: 0.833 — "The assistant provided the correct answer..."
  Builtin.Correctness [5b17ea26]: 1.0   — "The assistant correctly names the largest ocean..."
```

Full online chain, all real: AgentCore agent → auto OTEL export → CloudWatch →
SAES Logs-Insights discovery → native read-back → evaluators + Bedrock OpenAI
judge → EMF metrics + JSON results back to CloudWatch.

> NOTE: the deployed agent runtime `saesonline-*` and its log groups are AWS
> resources that incur cost. Destroy when done:
> `cd online_agent && agentcore destroy` and delete `/aws/saes/online-demo-results`.
