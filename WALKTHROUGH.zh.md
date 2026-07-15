> 英文版见 [WALKTHROUGH.md](WALKTHROUGH.md)。

# SAES — 完整演练：AgentCore Runtime → CloudWatch → 评估

一条线性路径，中途不跳步。你将：构建一个小型 agent，将其部署到 Amazon
Bedrock **AgentCore Runtime**，调用它（这会自动将 OTEL 追踪导出到
**CloudWatch**），然后让 **SAES** 读取这些追踪并对其打分。下面的每一条
命令都真实运行过；此处展示了确切的输出，方便你在每一步核对你的进度。

**开始之前你需要准备：**
- 一个 AWS 账户，在 `us-east-1` 中启用了 **Bedrock**（已开通模型访问权限）、
  **AgentCore** 和 **CloudWatch**，并已在本地配置好 AWS 凭证
  （`aws sts get-caller-identity` 应当能成功执行）。
- 已安装 **Python 3.12** 和 **Docker**。

本演练分为两部分：
- **Part A — 部署并运行你的 agent**（步骤 1–4）：在 CloudWatch 中生成追踪。
- **Part B — 用 SAES 评估**（步骤 5–6）：一条命令，只需 runtime id。

---

## Part A — 部署一个向 CloudWatch 发送追踪的 agent

### 步骤 1 — 安装 SAES 和 AgentCore 工具包

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

确认 CLI 可用：

```bash
saes --help          # ⇒ Commands: eval | run | doctor | init | serve
```

### 步骤 2 — 创建 agent（在一个新文件夹中放 3 个小文件）

这个 agent 是一个带两个工具的普通 Strands agent。AgentCore 的 runtime 已经
做了 OTEL 插桩，因此部署后它会自行将追踪导出到 CloudWatch —— 你**无需**
编写任何遥测代码。

> 想改用 **LangGraph / CrewAI / 无框架** 来构建？agent 的工作方式相同；
> 不同之处在于你的框架发出的 OTEL。参见
> [DOCUMENTATION.zh.md §7.4](DOCUMENTATION.zh.md#74-what-your-agent-must-emit--the-otel-contract-by-framework)
> 了解各框架的契约（scope 名称、必需字段、各评估器的需求）——
> 然后运行 `saes doctor` 在依赖分数之前进行验证。

```bash
mkdir -p my_agent && cd my_agent
```

`my_agent/agent.py`：

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

`my_agent/requirements.txt`：

```
bedrock-agentcore
strands-agents
```

`my_agent/Dockerfile`（AgentCore 运行在 ARM64 上；`aws-opentelemetry-distro`
这一行正是让追踪流向 CloudWatch 的关键）：

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

### 步骤 3 — 部署到 AgentCore Runtime（约 5 分钟，CodeBuild ARM64）

在 `my_agent/` 目录内：

```bash
export AGENTCORE_SUPPRESS_RECOMMENDATION=1
export BEDROCK_MODEL_ID="us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# configure (the printf feeds defaults to the interactive prompts)
printf '\n\n\n\n\n' | agentcore configure -e agent.py -n myagent -rf requirements.txt --create

# one-time gotcha: the toolkit defaults ECR auto-create to false — turn it on
sed -i 's/ecr_auto_create: false/ecr_auto_create: true/' .bedrock_agentcore.yaml

agentcore deploy -env BEDROCK_MODEL_ID="$BEDROCK_MODEL_ID"
```

完成后，记下它打印出的 **runtime id**（形如
`myagent-XXXXXXXXXX`）。它的日志组是：

```
/aws/bedrock-agentcore/runtimes/myagent-XXXXXXXXXX-DEFAULT
```

> 上面的文件已经处理好的坑：starter toolkit 有时不会写出 `Dockerfile`
> （你已经有了一个）；`ecr_auto_create` 默认为 false（`sed` 会把它翻转）。
> 如果 configure 询问关于 Dockerfile 的问题，保留你自己的。

### 步骤 4 — 调用 agent（这会生成你将要打分的追踪）

```bash
agentcore invoke '{"prompt": "What is the weather in Tokyo?"}'
agentcore invoke '{"prompt": "What is 15% of 240?"}'
agentcore invoke '{"prompt": "Weather in Paris, and what is 12*8?"}'
```

每条都会返回一个 `{"result": "..."}` 的答复。**现在等待约 90–100 秒** ——
追踪投递到 CloudWatch 再加上 Logs Insights 建立索引会有短暂延迟。这就是
Part A 的结尾：你的 agent 的 OTEL 追踪现在已经在 CloudWatch 中了。

---

## Part B — 用 SAES 评估这些 CloudWatch 追踪

回到仓库根目录（从 `my_agent/` 中 `cd ..` 出来），仍在已激活的 venv 中。

### 步骤 5 — 设置 judge 密钥

SAES 使用 LLM-as-a-Judge 打分。在 AWS 上最简单的方案是 Bedrock 的
OpenAI 兼容端点，用从你自己的 AWS 凭证铸造出的短期 token 进行认证
（无需外部 API key）：

```bash
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
```

> 该 token 约 12 小时后过期；如果遇到认证错误，重新运行这一行即可。

### 步骤 6 — 评估 —— 一条命令，只需 runtime id

把步骤 3 得到的 runtime id 交给 `saes eval`。它会推导出日志组、发现
会话（session）并对它们打分 —— **无需 YAML、无需 ground truth、无需手动
查找会话**：

```bash
saes eval myagent-XXXXXXXXXX --html out/report.html
```

这条命令的真实输出（针对参考用的 `saesstrands` runtime）：

```
evaluating /aws/bedrock-agentcore/runtimes/myagent-XXXXXXXXXX-DEFAULT
  judge: openai.gpt-oss-20b-1:0  |  last 7d  |  12 evaluator(s): Builtin.Helpfulness, Builtin.Coherence, ...

eval-myagent-XXXXXXXXXX  (judge: openai.gpt-oss-20b-1:0)  [1 session(s)]
  Builtin.Helpfulness              avg=0.833  pass=100%  n=1
  Builtin.Coherence                avg=1.000  pass=100%  n=1
  Builtin.Conciseness              avg=1.000  pass=100%  n=1
  Builtin.Faithfulness             avg=1.000  pass=100%  n=1
  Builtin.InstructionFollowing     avg=1.000  pass=100%  n=1
  Builtin.ResponseRelevance        avg=1.000  pass=100%  n=1
  Builtin.ContextRelevance         avg=1.000  pass=100%  n=1
  Builtin.Harmfulness              avg=1.000  pass=100%  n=1
  Builtin.Refusal                  avg=0.000  pass=0%  n=1
  Builtin.Stereotyping             avg=1.000  pass=100%  n=1
  Builtin.ToolSelectionAccuracy    avg=1.000  pass=100%  n=...
  Builtin.ToolParameterAccuracy    avg=1.000  pass=100%  n=...

HTML  → out/report.html
```

这就是完整的闭环：**AgentCore agent → 自动 OTEL → CloudWatch → `saes eval`
读取并打分。** `out/report.html` 是一份自包含的报告，里面有 judge 对每个
结果的推理。

### 选择评估器与采样（与 AgentCore 对齐的选项）

默认情况下 `saes eval` 运行 **12 个无需参考（reference-free）的内置评估器**
（它们无需 ground truth）。列出所有可用项、挑选你自己的，或运行全部 13 个：

```bash
saes eval --list-evaluators                       # show all built-in ids

saes eval myagent-XXXXXXXXXX \
  --evaluators Builtin.Helpfulness,Builtin.Harmfulness \  # only these (also: -e)
  --sampling 25 \                                         # score 25% of sessions (deterministic)
  --days 30 \                                             # scan the last 30 days
  --judge-model gpt-4.1 --judge-base-url https://api.openai.com/v1   # a different judge

saes eval myagent-XXXXXXXXXX --all                # all 13 built-ins
```

> 其中 **Correctness / GoalSuccessRate / Trajectory\*Match** 这三个评估器
> 会针对 ground truth（期望答案 / 断言 / 期望的工具序列）打分。`--all`
> 包含 Correctness + GoalSuccessRate，但只有在你提供 ground truth 时它们
> 才打得最好 —— 为此（以及 CI 门禁，或自定义 LLM/代码评估器），请编写
> 完整配置并使用 `saes run` / `saes serve` —— 见 DOCUMENTATION.zh.md §5、§6、§11。

如果你看到 `no sessions found`，说明该 agent 最近没有运行过，或者会话早于
时间窗口 —— 用 `--days 30` 拓宽它。

---

## 清理（完成后）

```bash
cd my_agent && AGENTCORE_SUPPRESS_RECOMMENDATION=1 agentcore destroy && cd ..
```

## 如果某一步行为异常

| 症状 | 解决办法 |
|---|---|
| `no sessions found` | 追踪尚未建立索引（调用后等约 90 秒），或会话早于时间窗口 —— 拓宽它：`--days 30`。 |
| `log group not found` | runtime id 有误，或该 agent 从未发出过追踪。使用裸 id（例如 `myagent-XyZ123`，不带 `-DEFAULT`）。 |
| judge 认证错误 | 重新运行步骤 5 的 export；Bedrock token 会过期（约 12 小时）。 |
| judge 探测 / 打分在非 Bedrock 端点上失败 | 你的端点必须支持工具调用 / 结构化输出。用 `saes doctor --judge <config>` 验证。 |
| `ModuleNotFoundError: openai` | `pip install openai`（`openai_compatible` judge 需要它）。 |

## 持续监控（在线）—— 同一条单行命令

上面的一切都是一次性的（`saes eval`）。要持续评估一个**运行中（live）**
agent 的新流量，用相同的 runtime id 运行 `saes serve`：

```bash
saes serve myagent-XXXXXXXXXX                 # polls every 60s, scores completed sessions
saes serve myagent-XXXXXXXXXX --once          # a single cycle (CI/cron)
```

它会在 `/aws/saes/<runtime>-results` 自动创建一个结果汇聚点（sink），并接受
相同的 `-e` / `--all` / `--sampling` / `--judge-model` 标志，外加
`--session-timeout`（会话被计为完成前的静默分钟数）。见 DOCUMENTATION.zh.md §9。

## 下一步去哪里

- **其他框架**（LangGraph / CrewAI / 一个纯粹的无框架脚本）以完全相同的
  方式评估 —— 只有它们发出 span 的方式不同；SAES 的摄取会自适应。
  见 DOCUMENTATION.zh.md §7（怎么做）和 §8（四个框架的结果）。
- **还没有 agent / 离线与 CI** —— 对捆绑的追踪样本或本地 OTLP 转储打分，
  而不是 CloudWatch：DOCUMENTATION.zh.md §4.0。
- **完整参考** —— 配置、评估器目录、在线 worker、调优：
  DOCUMENTATION.zh.md。
