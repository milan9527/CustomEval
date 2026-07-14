# SAES — Complete walkthrough: AgentCore Runtime → CloudWatch → evaluation

One linear path, no jumps. You will: build a small agent, deploy it to Amazon
Bedrock **AgentCore Runtime**, invoke it (which auto-exports OTEL traces to
**CloudWatch**), then have **SAES** read those traces and score them. Every
command below was run for real; the exact output is shown so you can check your
progress at each step.

**What you need before starting:**
- An AWS account with **Bedrock** (model access enabled), **AgentCore**, and
  **CloudWatch** in `us-east-1`, and AWS credentials configured locally
  (`aws sts get-caller-identity` should succeed).
- **Python 3.12** and **Docker** installed.

The walkthrough has two halves:
- **Part A — deploy & run your agent** (Steps 1–4): produces traces in CloudWatch.
- **Part B — evaluate with SAES** (Steps 5–6): one command, just the runtime id.

---

## Part A — Deploy an agent that emits traces to CloudWatch

### Step 1 — Install SAES and the AgentCore toolkit

```bash
git clone https://github.com/milan9527/CustomEval.git
cd CustomEval
python3.12 -m venv .venv
source .venv/bin/activate                         # ← activate FIRST; use this venv throughout
pip install --upgrade pip
pip install -e '.[dev]' openai                    # SAES (the `saes` command)
pip install bedrock-agentcore bedrock-agentcore-starter-toolkit strands-agents \
            aws-bedrock-token-generator            # to build/deploy the agent + mint a judge token
```

Confirm the CLI is available:

```bash
saes --help          # ⇒ Commands: run | doctor | init | serve
```

### Step 2 — Create the agent (3 small files in a new folder)

The agent is a normal Strands agent with two tools. AgentCore's runtime is
OTEL-instrumented, so once deployed it exports traces to CloudWatch on its own —
you add **no** telemetry code.

```bash
mkdir -p my_agent && cd my_agent
```

`my_agent/agent.py`:

```python
"""A Strands tool-calling agent on AgentCore. Native OTEL -> CloudWatch."""
import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel

@tool
def get_weather(city: str) -> str:
    """Get the current weather forecast for a city."""
    return f"{city}: 22C, partly cloudy, light wind."

@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '15/100*240'."""
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
           ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}
    def ev(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp): return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp): return ops[type(n.op)](ev(n.operand))
        raise ValueError("bad expr")
    return str(ev(ast.parse(expression, mode="eval").body))

app = BedrockAgentCoreApp()
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
_agent = Agent(model=BedrockModel(model_id=MODEL_ID), tools=[get_weather, calculate],
               system_prompt="You are a helpful assistant. Use tools for weather and math.")

@app.entrypoint
def invoke(payload):
    return {"result": str(_agent(payload.get("prompt", "Hello")))}

if __name__ == "__main__":
    app.run()
```

`my_agent/requirements.txt`:

```
bedrock-agentcore
strands-agents
```

`my_agent/Dockerfile` (AgentCore runs ARM64; the `aws-opentelemetry-distro` line
is what makes traces flow to CloudWatch):

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_SYSTEM_PYTHON=1 UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
COPY requirements.txt requirements.txt
RUN uv pip install -r requirements.txt
RUN uv pip install aws-opentelemetry-distro==0.12.2
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore
EXPOSE 8080
COPY . .
CMD ["opentelemetry-instrument", "python", "-m", "agent"]
```

### Step 3 — Deploy to AgentCore Runtime (~5 min, CodeBuild ARM64)

From inside `my_agent/`:

```bash
export AGENTCORE_SUPPRESS_RECOMMENDATION=1
export BEDROCK_MODEL_ID="us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# configure (the printf feeds defaults to the interactive prompts)
printf '\n\n\n\n\n' | agentcore configure -e agent.py -n myagent -rf requirements.txt --create

# one-time gotcha: the toolkit defaults ECR auto-create to false — turn it on
sed -i 's/ecr_auto_create: false/ecr_auto_create: true/' .bedrock_agentcore.yaml

agentcore deploy -env BEDROCK_MODEL_ID="$BEDROCK_MODEL_ID"
```

When it finishes, note the **runtime id** it prints (looks like
`myagent-XXXXXXXXXX`). Its log group is:

```
/aws/bedrock-agentcore/runtimes/myagent-XXXXXXXXXX-DEFAULT
```

> Gotchas already handled by the files above: the starter toolkit sometimes
> doesn't write a `Dockerfile` (you have one); `ecr_auto_create` defaults to
> false (the `sed` flips it). If configure asks about a Dockerfile, keep yours.

### Step 4 — Invoke the agent (this creates the traces you'll score)

```bash
agentcore invoke '{"prompt": "What is the weather in Tokyo?"}'
agentcore invoke '{"prompt": "What is 15% of 240?"}'
agentcore invoke '{"prompt": "Weather in Paris, and what is 12*8?"}'
```

Each returns a `{"result": "..."}` answer. **Now wait ~90–100 seconds** — trace
delivery to CloudWatch plus Logs Insights indexing has a short lag. That's the
end of Part A: your agent's OTEL traces are now in CloudWatch.

---

## Part B — Evaluate those CloudWatch traces with SAES

Back in the repo root (`cd ..` out of `my_agent/`), still in the activated venv.

### Step 5 — Set the judge key

SAES scores with an LLM-as-a-Judge. The simplest option on AWS is Bedrock's
OpenAI-compatible endpoint, authenticated with a short-term token minted from
your own AWS credentials (no external API key):

```bash
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region=\"us-east-1\"))')"
```

> The token expires in ~12h; just re-run this line if you get an auth error.

### Step 6 — Evaluate — one command, just the runtime id

Give `saes eval` the runtime id from Step 3. It derives the log group, discovers
the sessions, and scores them — **no YAML, no ground truth, no manual session
lookup**:

```bash
saes eval myagent-XXXXXXXXXX --html out/report.html
```

Real output from this exact command (against the reference `saesstrands` runtime):

```
evaluating /aws/bedrock-agentcore/runtimes/myagent-XXXXXXXXXX-DEFAULT
  judge: openai.gpt-oss-20b-1:0  |  evaluators: Builtin.Helpfulness, Builtin.Coherence, Builtin.Conciseness, Builtin.ResponseRelevance

eval-myagent-XXXXXXXXXX  (judge: openai.gpt-oss-20b-1:0)
  Builtin.Helpfulness              avg=0.833  pass=100%  n=1
  Builtin.Coherence                avg=1.000  pass=100%  n=1
  Builtin.Conciseness              avg=1.000  pass=100%  n=1
  Builtin.ResponseRelevance        avg=1.000  pass=100%  n=1

HTML  → out/report.html
```

That's the full loop: **AgentCore agent → auto OTEL → CloudWatch → `saes eval`
reads + scores it.** `out/report.html` is a self-contained report with the
judge's reasoning per result.

Useful flags (all optional):

```bash
saes eval myagent-XXXXXXXXXX \
  --lookback-days 3 \                       # if your session is older than a day
  --evaluators Builtin.Helpfulness,Builtin.Coherence \   # pick your own
  --judge-model gpt-4.1 --judge-base-url https://api.openai.com/v1   # a different judge
```

> `saes eval` uses **reference-free** evaluators by default (Helpfulness,
> Coherence, Conciseness, ResponseRelevance) — they need no ground truth. To also
> score against expected answers or tool trajectories (Correctness,
> Trajectory*Match, ToolSelectionAccuracy…), or to gate a CI build, write a full
> config and use `saes run` / `saes serve` — see DOCUMENTATION.md §5, §11.

---

## Cleanup (when you're done)

```bash
cd my_agent && AGENTCORE_SUPPRESS_RECOMMENDATION=1 agentcore destroy && cd ..
```

## If a step doesn't behave

| Symptom | Fix |
|---|---|
| `no sessions scored` | Traces not indexed yet (wait ~90s after invoking), or the session is older than `--lookback-days` — increase it (e.g. `--lookback-days 3`). |
| A judge auth error | Re-run the Step 5 export; the Bedrock token expires (~12h). |
| Judge probe / scoring fails on a non-Bedrock endpoint | Your endpoint must support tool calling / structured output. Verify with `saes doctor --judge <config>`. |
| `ModuleNotFoundError: openai` | `pip install openai` (needed for the `openai_compatible` judge). |

## Where to go next

- **Other frameworks** (LangGraph / CrewAI / a plain no-framework script) evaluate
  the exact same way — only how they emit spans differs; SAES's ingestion adapts.
  See DOCUMENTATION.md §7 (how) and §8 (the four-framework results).
- **No agent yet / offline & CI** — score a bundled trace sample or a local OTLP
  dump instead of CloudWatch: DOCUMENTATION.md §4.0.
- **Full reference** — configuration, evaluator catalog, online worker, tuning:
  DOCUMENTATION.md.
