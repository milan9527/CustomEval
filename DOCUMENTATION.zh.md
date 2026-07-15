> 英文版见 [DOCUMENTATION.md](DOCUMENTATION.md)。

# SAES — Strands Agent Evaluation Suite

**面向 AI agent 的开源评估方案，基于 [Strands Agents SDK](https://strandsagents.com/) 构建，并与 Amazon Bedrock AgentCore Observability 集成。**

这是 SAES 唯一、统一的参考文档。它涵盖了本项目是什么、如何构建、如何端到端使用（从构建 agent 到生成评分报告）、它验证过的框架与评估场景，以及对真实结果的分析。

> 本文档取代此前分散的各篇文档（README、ARCHITECTURE、USAGE、
> VERIFICATION、FRAMEWORK_MATRIX、BUILTIN_SUITE、BAD_EXAMPLES、RUN_LOG、
> REPRODUCE、MULTIFRAMEWORK_RESULTS）。完整的技术规范仍保留在
> [SPEC.md](SPEC.md)。

---

## 目录

1. [什么是 SAES](#1-什么是-saes)
2. [两大差异化特性](#2-两大差异化特性)
3. [架构与实现](#3-架构与实现)
4. [端到端使用：从构建 agent 到生成评分报告](#4-端到端使用从构建-agent-到生成评分报告)
   — 从 [§4.0 “我刚克隆了这个仓库，也有自己的 agent”](#40-我刚克隆了这个仓库也有自己的-agent--从哪里开始) 开始
5. [配置参考](#5-配置参考)
6. [评估器目录](#6-评估器目录)
7. [框架支持：任意框架如何达到完整覆盖](#7-框架支持任意框架如何达到完整覆盖)
   — 含 [§7.4 你的 agent 必须发射什么（按框架划分的 OTEL 契约）](#74-你的-agent-必须发射什么--按框架划分的-otel-契约)
8. [评估场景与结果分析](#8-评估场景与结果分析)
   — 含 [§8.5 逐步拆解评估过程](#85-逐步拆解评估过程)
9. [在线/生产环境评估](#9-在线生产环境评估)
10. [验证记录：已证实的能力与发现的 bug](#10-验证记录已证实的能力与发现的-bug)
11. [复现](#11-复现)
12. [故障排查](#12-故障排查)

---

## 1. 什么是 SAES

SAES 是一套可自托管的 AI agent 评估方案。它读取你的 agent 已经在发射的
OpenTelemetry（OTEL）trace，重建每一次对话，并用一整套评估器（LLM-as-a-Judge 与
确定性两类）为其打分——这些评估器对齐了
[Amazon Bedrock AgentCore Evaluations](https://aws.amazon.com/blogs/machine-learning/build-reliable-ai-agents-with-amazon-bedrock-agentcore-evaluations/)。
它可以离线运行（读取本地 trace dump，用于 CI/回归），也可以在线运行（对线上 agent
的 CloudWatch 流量进行采样），并把结果作为指标 + 结构化日志写回 CloudWatch，使质量信号
与运维信号并列呈现。

**设计立场——复用而非重造。** 评估*引擎*就是原生的
[`strands-agents-evals`](https://strandsagents.com/)：内置评估器、trajectory 评分器、
prompt 模板、session mapper、trace provider，以及
`Experiment`/`Case`/`Report` 编排。SAES 只添加该包所缺少的几个薄层：

1. 将 **OpenAI 兼容的评审模型（judge）选择** 作为一等的配置面。
2. 一个 **框架无关的 CloudWatch/OTEL 采集适配器**（正是它让*任何* agent 都可被评估，
   而不仅限于 Strands）。
3. 一个 **在线采样 worker** + CloudWatch 结果发射。

SAES 从不导入或运行你的 agent 代码。唯一的集成契约就是 trace 格式。

### 它不是什么

- 它不是 AgentCore Observability 遥测管线的替代品——它*消费*那条管线。
- 它不是托管服务——它是一个库 + CLI + 可选 worker。
- 它不是 agent 编写框架——它评估 agent，而不构建 agent。
- 它不局限于 Strands agent——名字反映的是它*构建所用*的 SDK，而非它*能评估*的对象。

### 状态

M1（核心离线评估）、M2（CloudWatch 采集、EMF/JSON 结果、完整评估器目录）以及 M3
（在线 worker、Lambda 代码评估器、dashboard/告警 CDK）均已完成。**199 个单元测试通过**
（外加 CDK synth 测试）。已用真实 Bedrock judge 完成端到端验证（离线与在线），针对真实部署的
AgentCore Runtime agent，并跨四种框架（Strands、LangGraph、CrewAI、无框架）验证。Apache-2.0。
尚未发布。

---

## 2. 两大差异化特性

### 2.1 自带 judge（Bring your own judge）

LLM-as-a-Judge 可以是**任何支持工具调用/结构化输出的 OpenAI 兼容端点**，也可以是
Amazon Bedrock。这意味着 OpenAI、Azure OpenAI、自托管 vLLM（引导式解码）、LiteLLM、
SageMaker，或 Bedrock 的 OpenAI 兼容 API——你不会被锁定在某个托管 judge 上。

> **硬性要求：** 原生评估器通过
> `invoke_async(prompt, structured_output_model=...)` 打分——它们需要经由工具调用获得
> 结构化输出，而非自由文本。一个只支持文本的 chat-completions 端点会以
> `StructuredOutputException` 失败。SAES 通过一次**预检探测**
> （`saes doctor --judge`）强制这一点，从而在开始前就用可操作的提示拒绝坏端点，
> 绝不会在运行中途莫名崩溃。

已验证的 judge 包括 Bedrock（原生 + OpenAI 兼容），以及——经由
Bedrock OpenAI API——DeepSeek、Kimi 和 Qwen（见 §10）。

### 2.2 框架无关的输入

任何 agent——任意框架、任意语言——只需向 CloudWatch（或本地 dump）发射符合 OTEL GenAI
约定的 trace，即可被评估。SAES 的采集层会适配每种框架自然发射的各类 span，并从中重建出
一个统一的评估轮次（turn）。**适配逻辑存在于 SAES 采集层，而非 agent 中。** 一个零 SAES
专用埋点的裸 `boto3` 脚本，能获得与原生 Strands agent 相同的评估器覆盖（如何做到见 §7，
跨四种框架的证据见 §8）。

---

## 3. 架构与实现

### 3.1 SAES 自有 vs. 复用

| 关注点 | SAES 自有（本仓库） | 原生 `strands-agents-evals` |
|---|---|---|
| 配置 schema / CLI | ✅ `config/`、`cli.py` | — |
| judge 选择 + 探测 | ✅ `judge/` | 模型提供方（`strands.models`） |
| 评估器解析（ids → 原生、自定义 LLM/代码、trajectory） | ✅ `evaluators/` | 评估器类本身 |
| trace 采集（factory、本地 reader、CloudWatch 发现 + 补全） | ✅ `ingest/` | providers + 会话映射器（session mappers） |
| 运行编排接线 + 聚合 + 门禁（gate） | ✅ `run/` | `Experiment.run_evaluations_async` |
| 报告（JSON/HTML）+ CloudWatch EMF sink | ✅ `report/` | — |
| 在线 worker（发现 → 完成 → 采样 → 打分 → 发射） | ✅ `online/` | 它所驱动的评分流水线 |
| 评估器、模板、评分、生成、检测器 | — | ✅ |

### 3.2 模块图

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

### 3.3 按需评估流程（`saes run`）

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

关键接线事实（已针对真实 SDK 验证）：
- 原生报告的 `detailed_results` 是**以评估器为主维度并已扁平化的**——每个（评估器，case）
  一行。聚合以 `report.cases[i]["evaluator"]` 为键，SAES 将其设为 AgentCore 风格的 id。
- 原生评估器**以其 id 命名**，因此同一个类可以在一次 Experiment 中多次出现（引擎会拒绝
  重复的名字）。
- judge 是一个原生的 strands `Model`；SAES 通过 `model=` 注入它。

### 3.4 采集接缝（框架无关）

`ingest.load_sessions(cfg)` 返回原生 `Session` 对象；在正常路径下 SAES 不编写自己的映射
代码：

- **`otlp_file`**（离线/CI）：读取本地 JSONL/OTLP dump，按 session id 分组 span，把每组交给
  `detect_otel_mapper()`。适用于 CloudWatch / OpenInference / LangChain-OTEL 的 dict 格式。
- **`cloudwatch`**（生产）：运行一次 Logs Insights 查询以**发现 session id**（原生 provider
  只能按已知 id 读取），然后把每个 session 的读取+映射委托给原生
  `CloudWatchProvider`。当原生 mapper 无法重建某框架的 span 时，SAES 的**补全**（§7）填补
  空缺。
- **`live`**：为正在运行的 Strands agent 做原生内存 span 捕获。

`saes doctor --data-source` 会报告逐字段的覆盖情况，从而在运行*之前*就暴露缺口。

---

## 4. 端到端使用：从构建 agent 到生成评分报告

这就是完整旅程。自上而下跟着走，即可完成第一次可用的评估。

### 4.0 “我刚克隆了这个仓库，也有自己的 agent——从哪里开始？”

> **想要一个从头到尾的完整单例？** [WALKTHROUGH.zh.md](WALKTHROUGH.zh.md)
> 是一条线性路径——克隆 → 构建 agent → 部署到 AgentCore Runtime →
> trace 进入 CloudWatch → SAES 打分——没有任何跳跃，且给出每条命令的
> 真实输出。本节则是较短的“哪条路径适合我”概览，以及离线/立刻试用的选项。

你无需修改 agent，也无需改动 SAES 的源码。SAES 是一个你指向 agent *已经*产生的 trace 的
工具：安装 CLI → 把 trace 放到 SAES 能读取的地方 → 写一小段 YAML → 运行。

下面每条命令都是在一份干净克隆上运行的；输出原样展示，你便知道“正常”是什么样子。

#### 第 1 步——安装（所有人相同）

**前置条件：Python 3.12**（venv 必须使用它——3.9/3.10 的系统 `python3` 也能安装，但经过
验证的是 3.12）。先用 `python3.12 --version` 检查；若缺失则安装
（`sudo dnf install python3.12` / `apt install python3.12` / `brew install python@3.12`）。

```bash
git clone https://github.com/milan9527/CustomEval.git
cd CustomEval
python3.12 -m venv .venv
source .venv/bin/activate                 # ← activate FIRST; run everything below inside it
pip install --upgrade pip
pip install -e '.[dev]' openai            # installs the `saes` command + all deps
saes --help                               # ⇒ Commands: eval | run | doctor | init | serve
```

> 如果 `pip install` 报 `No matching distribution found for
> strands-agents`，说明你没有处于已激活的 venv，或者你的 pip 指向了私有索引——先激活，
> 或用 `--index-url https://pypi.org/simple/` 强制走公共 PyPI。

#### 第 2 步——用仓库自带的 trace 样本证明它能跑通（约 1 分钟）

在接入自己的 agent 之前，先确认整条链路能工作。仓库自带了一份真实的 trace fixture，可以
立刻打分。你只需要一个 judge——这里用 Amazon Bedrock（使用你的 AWS 凭证，无需外部 API
key）：

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

预期的最终输出（已验证）：

```
try-it  (judge: openai.gpt-oss-20b-1:0)
  Builtin.Helpfulness              avg=0.833  pass=100%  n=1

JSON  → out/results.json
HTML  → out/report.html
```

如果你看到上述内容，说明一切正常，`out/report.html` 中就有 judge 的推理过程。现在把它
指向你自己的 agent。

> 完全没有 AWS？改在 `judge` 块里用任意支持工具调用的 OpenAI 兼容端点即可（设置
> `base_url`/`model`，并把 key 放进 `SAES_JUDGE_API_KEY`）。只支持文本的端点行不通——
> `saes doctor --judge` 会提前告诉你。

#### 第 3 步——把它指向*你自己* agent 的 trace（选择你的路径）

你**不需要** Strands，也**不需要**添加任何 SAES 专用遥测——SAES 的采集会适配你的框架
发射的任意标准 OTEL（§7）。唯一的契约是“按 `session.id` 分组的 span”。按你手头拥有的
条件选择：

**路径 A——你的 agent 在 AgentCore Runtime 上**（trace 自动导出到 CloudWatch）。
只需把 runtime id 交给 `saes eval`——它会推导出 log group、发现各 session 并为其打分。
无需 YAML，无需 ground truth：

```bash
saes eval <your-runtime-id> --html out/report.html
#   scans the last 7 days by default; widen with --days 30 if the session is older
```

各选项（对齐 AgentCore Evaluations）：

```bash
saes eval --list-evaluators                          # show all built-in ids
saes eval <runtime> --all                            # all 13 built-ins (default: 12 reference-free)
saes eval <runtime> -e Builtin.Helpfulness,Builtin.Harmfulness   # choose evaluators
saes eval <runtime> --sampling 25                    # score 25% of sessions (deterministic)
saes eval <runtime> --judge-model gpt-4.1 --judge-base-url https://api.openai.com/v1
```

这就是 AgentCore 的全部。（对于三个需要 ground truth 的评估器——Correctness /
GoalSuccessRate / Trajectory\*Match——或者 CI 门禁，或者自定义 LLM/代码评估器，请写一份
完整配置并使用 `saes run` / `saes serve`——§5、§6、§11。要指向你自己的非 AgentCore
CloudWatch log group，请用带 `dataSource.type: cloudwatch` 和 `log_group_names` 的配置。）

**路径 B——你能导出本地 OTEL/OTLP dump**（开发/CI，无 trace 存储）。
把你的 span 保存到一个 JSONL 文件（每行一条 span 记录），然后：

```yaml
dataSource:
  type: otlp_file
  path: ./my_traces.jsonl
```

```bash
saes doctor --data-source ./my_traces.jsonl      # ← ALWAYS run this first (see below)
saes run   -c try.yaml --html out/report.html
```

#### 能救你的那个习惯：先跑 `saes doctor`

在信任分数之前，先运行 `saes doctor --data-source <your dump>`（对于 CloudWatch，则确认
某个 session 被发现——§8.5 第 1 步）。它会打印逐字段覆盖情况，以及你的 trace 是否重建为
**可评估的** session：

```
spans read: 4
field coverage:
  ✓ session id            4/4
  ✓ prompt / input        4/4
  ✓ completion / output   4/4
  ✗ tool name             0/4   (expected if this agent uses no tools)
OK — 2 session(s) reconstructed
```

session id / prompt / completion 上出现 `✗`，说明你的埋点缺少这些 GenAI 属性——请在源头
修复，否则会得到空分数（`n=0`）。注意：Strands scope 的**本地 dump** 不能从文件往返
（对 Strands 请使用 CloudWatch 源或内存路径）；CloudWatch / OpenInference /
LangChain-OTEL 的 dump 可以从文件工作。见 §10（F4）。

§4 的其余部分是同一旅程的完整细节；§5 是配置参考，§8.5 是每次运行执行的确切流水线。

### 第 1 步——拥有一个发射 OTEL trace 的 agent

SAES 从 agent 的 OpenTelemetry trace 中评估它；它从不运行你的代码。你唯一要做的，是让
agent **发射符合 OTEL GenAI 约定的 span**，并按 `session.id` 分组。三种常见情形：

- **Strands / AgentCore Runtime agent**——免费。AgentCore 的 runtime 已做 OTEL 埋点，并
  自动导出到 CloudWatch。
- **LangGraph / CrewAI / 其他框架**——启用其 OpenTelemetry / OpenInference 埋点；在
  AgentCore 上这会自动导出，或自托管一个 ADOT collector。
- **完全无框架**——一个纯脚本的 Bedrock 调用会被 AgentCore 的 botocore 埋点捕获；SAES 从
  这些标准 span 中重建轮次（见 §7）。无需任何 SAES 专用代码。

### 第 2 步——运行你的 agent 以产生 trace

用有代表性的输入去驱动 agent。这会产生 SAES 要打分的 trace——要么在 CloudWatch log group
中（生产/在线），要么是本地 OTLP/JSONL dump（离线/CI）。

### 第 3 步——安装 SAES 并验证你的 trace

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # activate FIRST
pip install -e '.[dev]' openai
saes doctor --data-source traces.jsonl                  # offline dump
```

`doctor` 会报告逐字段覆盖（session id、prompt/completion、tool name……）以及 session 是否
重建成功。继续之前请先修复任何 ✗。

> 只要 `judge.provider: openai_compatible`，就必须安装 `openai` 包。对于
> `judge.provider: bedrock`，你只需要 AWS 凭证。如果 `pip install`
> 报 `No matching distribution found for strands-agents`，很可能你不在已激活的 venv 中，
> 或你的 pip 指向了私有索引——请激活 venv，或用
> `--index-url https://pypi.org/simple/` 强制走公共 PyPI。

### 第 4 步——选一个 judge 并验证它合格

```bash
export SAES_JUDGE_API_KEY=...              # or a Bedrock bearer token (§5.2)
saes doctor --judge eval.yaml              # → ✓ structured output confirmed
```

只支持文本的端点会在此被拒绝，在运行之前。

### 第 5 步——写配置

```bash
saes init --agent-type rag --out eval.yaml   # scaffold with recommended evaluators
```

然后编辑 `eval.yaml`：把 `dataSource` 指向你的 trace，设置 `judge`，选择
`evaluators`，并可选地加上 `groundTruth` 和一个 CI `gate`。完整参考见 §5。

### 第 6 步——评估

```bash
saes run -c eval.yaml --json out/results.json --html out/report.html
```

控制台显示逐评估器分数；HTML 报告中含每条结果的 judge 推理。若某个 `gate` 阈值失败则以
非零退出（可接入 CI）。

### 第 7 步——（可选）生产监控

```bash
saes serve -c online.yaml --interval 60      # continuous; samples completed sessions
```

见 §9。

### 整个循环，最简版

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

## 5. 配置参考

### 5.1 `eval.yaml`

```yaml
name: my-agent-quality
mode: on_demand                      # on_demand | online

dataSource:
  type: otlp_file                    # otlp_file (local) | cloudwatch
  path: ./traces.jsonl               # for otlp_file
  # for cloudwatch instead:
  # type: cloudwatch
  # cloudwatch:
  #   log_group_names: ["/aws/bedrock-agentcore/runtimes/<agent>-DEFAULT"]
  #   agent_name: my-agent           # or discover the log group by name
  #   region: us-east-1
  #   lookback_days: 7

judge:
  provider: openai_compatible        # openai_compatible | bedrock
  model: "gpt-4.1"
  base_url: "https://your-endpoint/v1"   # required for openai_compatible
  api_key_env: "SAES_JUDGE_API_KEY"      # env var NAME (never the key itself)
  params: {temperature: 0.0}
  # provider: bedrock
  # model: "us.anthropic.claude-sonnet-4-5-..."   # no base_url/api_key needed

evaluators:
  - Builtin.Helpfulness
  - Builtin.Correctness              # uses expectedResponse when present
  - Builtin.ToolSelectionAccuracy
  - Builtin.GoalSuccessRate          # uses assertions
  - Builtin.TrajectoryInOrderMatch   # deterministic; uses expectedTrajectory
  - id: hipaa_compliance             # custom LLM evaluator
    type: llm
    level: trace
    instructions: |
      Score 1.0 if no PHI was disclosed without authorization, else 0.0.

groundTruth:
  path: ./ground_truth.jsonl         # optional

gate:                                # optional CI thresholds
  - "Builtin.Helpfulness.avg >= 0.8"
  - "Builtin.Correctness.avg >= 0.9"
```

**密钥：** API key 从 `api_key_env` 所指定的环境变量名中读取，绝不会存储在模型上或被
序列化。

### 5.2 用 Amazon Bedrock 作为 OpenAI 兼容 judge（已验证）

```yaml
judge:
  provider: openai_compatible
  model: "openai.gpt-oss-20b-1:0"        # or another Bedrock OpenAI model
  base_url: "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
  api_key_env: "SAES_JUDGE_API_KEY"
  params: {temperature: 0.0}
```

从环境中的 AWS 凭证生成一个短期 bearer token（继承你的 IAM role，自动过期）：

```bash
pip install aws-bedrock-token-generator
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
saes doctor --judge eval.yaml      # → ✓ structured output confirmed
```

> 也可以使用 `provider: bedrock`（原生 AWS 凭证，无需 token）。OpenAI 兼容路线的好处是能
> 用一份统一配置覆盖多个 provider。两者都经过端到端验证。

### 5.3 Ground truth（可选）

JSONL，每个 session 一条记录，以 `sessionId` 为键。每个评估器只读取它需要的字段：

```json
{"sessionId": "s-123", "expectedResponse": "You have 40 hours of PTO.",
 "assertions": ["Agent retrieved the balance from the HR system"],
 "expectedTrajectory": ["lookup_employee", "get_pto_balance"]}
```

- `expectedResponse` → `Builtin.Correctness`
- `assertions` → `Builtin.GoalSuccessRate`
- `expectedTrajectory` → `Builtin.Trajectory*Match`

---

## 6. 评估器目录

所有评估器底层都是原生的 `strands-agents-evals` 类，因此分数与托管的 AgentCore
Evaluations 对齐。

| 评估器 | 级别 | 是否需要 ground truth | 类型 |
|---|---|---|---|
| `Builtin.GoalSuccessRate` | Session | `assertions`（可选） | LLM |
| `Builtin.Helpfulness` | Trace | — | LLM |
| `Builtin.Correctness` | Trace | `expectedResponse`（可选） | LLM |
| `Builtin.Coherence` / `Conciseness` / `Faithfulness` | Trace | — | LLM |
| `Builtin.Harmfulness` / `Refusal` / `Stereotyping` | Trace | — | LLM |
| `Builtin.InstructionFollowing` / `ResponseRelevance` / `ContextRelevance`\* | Trace | — | LLM |
| `Builtin.ToolSelectionAccuracy` / `ToolParameterAccuracy` | Tool | — | LLM |
| `Builtin.TrajectoryExactOrderMatch` / `InOrderMatch` / `AnyOrderMatch` | Tool | `expectedTrajectory` | 确定性 |

\* 在当前 SDK 中 `ContextRelevance` 被别名为 ResponseRelevance（v1.0.2 中没有独立的原生
类）。

**明细：** 12 个纯 LLM-as-judge（无需参考答案）；`Correctness` +
`GoalSuccessRate` 是 LLM-judge，*可选*搭配 ground truth；3 个 trajectory 匹配器是
**确定性的**（不用 LLM，使用 `expectedTrajectory`）。这就是“13 个 AgentCore 内置评估器 +
ContextRelevance 别名 + 3 个 trajectory 评分器”。

### 自定义评估器（与 AgentCore 对齐）

- **LLM**——`type: llm` + `instructions`（一份评分标准 rubric）。使用你的 judge；包装原生
  `OutputEvaluator`。
- **代码**——一个确定性函数，由 `type: code` 引用。同一份函数体既能在本地运行，也能在生产
  中作为 Lambda 运行（M3）：

```python
from saes.evaluators import code_evaluator, CodeVerdict

@code_evaluator(id="paystub_amount", level="trace")
def check(case) -> CodeVerdict:
    ok = "$8,333.33" in str(case.actual_output)
    return CodeVerdict(1.0 if ok else 0.0, "PASS" if ok else "FAIL")
```

---

## 7. 框架支持：任意框架如何达到完整覆盖

核心设计目标：**任意框架，或无框架，都能被完整评估**——适配逻辑在 SAES 采集层，而非
agent 中。本节解释其机制；§8 展示证据。

### 7.1 问题所在

原生 `strands-agents-evals` mapper 是针对 Strands 发射的确切 OTEL span 形态调优的
（`AgentInvocationSpan` + `ToolExecutionSpan`）。其他框架发射的形态不同：

- **LangGraph**——原生读取成功，但只产出 `InferenceSpan`（没有 agent/tool span）。
- **无框架 / CrewAI**——原生 mapper *什么都重建不出来*，且读取*抛出*
  `SessionNotFoundError`。

若止步于此，就只有 Strands 能获得完整覆盖。但在所有情形中数据其实都在——只是处于原生
mapper 读不懂的形态。

### 7.2 SAES 的三项补全

SAES 的采集层会从每种框架发射的任意标准 OTEL 中重建出一个统一的评估轮次：

- **工具补全**（`ingest/tool_supplement`）：从原始 Converse `toolUse`/`toolResult` span 中
  恢复工具轨迹（名称/参数/结果），并打通 `trace_id → session.id`（botocore 工具 span 不
  携带 session id；同一 trace 中的某个 OpenInference span 携带）。
- **role 感知的轮次恢复**（`_iter_role_texts`）：从带 role 的 botocore Bedrock span 中恢复
  用户 prompt 与**最终答案**——`body.message`（role=assistant，
  `finish_reason=end_turn`）和 `body.{input,output}.messages`。
  *最终答案由 AgentCore 的 botocore 埋点捕获；真正的修复是正确地读取它，而非改动 agent。*
- **轮次 + tool span 合成**（`cloudwatch_task.supplement_turns`）：从恢复出的轮次构建原生
  `AgentInvocationSpan`（带 `available_tools`）和 `ToolExecutionSpan`——正是原生
  `TraceExtractor` 在 TRACE / SESSION / TOOL 级别所消费的确切形态。正是它让两个 tool 级别的
  LLM 评估器能对非 Strands agent 运行。
- **逐轮次重建**（`tool_supplement._reconstruct_turns`）：对于一个*多轮*
  session，它按 `trace_id`（一个 AgentCore trace = 一个轮次）把恢复出的文本 + 工具分组，
  按时间排序轮次，并**每轮合成一个 `AgentInvocationSpan`**——从而每一轮的 prompt 与该轮
  自己的答案配对，而不是与混淆的最后一个答案配对。没有这一步，一个 3 轮 session 会错配
  （例如把第 3 轮的“Paris?”与第 1 轮的“Tokyo”答案配对），从而给出错误分数。

所有补全都是尽力而为，绝不会向运行抛出异常；它们在 `saes run` 和 `saes serve` 中都会
自动生效。

### 7.3 结果：统一覆盖

有了这些补全，**全部 15 个内置评估器都能对全部四种框架运行。** 剩下的差异是由每个 agent
的实际行为驱动的*分数*差异——这恰恰是一套评估方案应当暴露的东西。有一点保真度说明：对于
非 Strands agent，合成的 `available_tools` 只携带工具*名称*（原始 span 不包含工具描述 /
JSON schema），因此 tool 级别的评估器是基于名称 + 观察到的调用来推理，而非完整的工具规格。

### 7.4 你的 agent 必须发射什么——按框架划分的 OTEL 契约

你不用写映射代码，但你的 trace 必须携带几样东西，SAES（以及底层的原生
`strands-agents-evals` mapper）才能重建出可评估的 session。这是你在每种框架里构建 agent
时要遵循的清单。**在依赖分数之前，务必用 `saes doctor --data-source <dump>` 验证**——它会
精确打印这些字段中哪些存在。

#### 通用契约（每种框架）

1. **span 上有 `session.id`**（可接受的键：`session.id`、
   `gen_ai.session.id` 或 `session_id`）。这是 span 如何分组为一次对话的依据。在 AgentCore
   Runtime 上，你通过给 `agentcore invoke` 传 `--session-id` 得到它；跨轮次复用 →
   一个多轮 session。
2. **Prompt/输入文本**——以下任意其一：`gen_ai.prompt`（或带索引的
   `gen_ai.prompt.N.content`）、`gen_ai.input.messages`、`input.value`、
   `llm.input_messages.*` 或 `traceloop.entity.input`。
3. **Completion/输出文本**——以下任意其一：`gen_ai.completion`（或
   `gen_ai.completion.N.content`）、`gen_ai.output.messages`、`output.value`、
   `llm.output_messages.*` 或 `traceloop.entity.output`。
4. **每个 span 上有 `scope.name`**——它决定选用哪个 mapper（见下文）。
5. **`traceId` + `spanId`**——标准 OTEL；**每轮一个 trace**（SAES 按 trace 对多轮 session
   分组，并按 span 时间排序）。

重建*任何东西*的最低要求：`session.id` **加上** prompt **或** completion。
Tool 级别和 trace 级别的评估器需要更多（见下文）。

#### scope 名称决定运行哪个 mapper

原生 mapper 由每个 span 的 `scope.name` 选取。原生只识别三个值：

| `scope.name` | 原生 mapper | 典型来源 |
|---|---|---|
| `strands.telemetry.tracer` | Strands mapper（完整：agent + tool span） | Strands SDK |
| `opentelemetry.instrumentation.langchain` | LangChain-OTEL mapper | 通过 OTEL instrumentor 的 LangChain/LangGraph |
| `openinference.instrumentation.langchain` | OpenInference mapper | OpenInference LangChain instrumentor |
| 其他任意值（`…crewai`、`botocore…`、自定义） | **无匹配** → 原生读取可能抛异常 | CrewAI、纯 boto3、自定义 |

**如果你的 scope 不在这三个之列，你并没有出错**——SAES 的补全（§7.2）会从原始 Bedrock
Converse span（`botocore` 的 `toolUse`/`toolResult`、带 role 的 `body.message`）恢复出轨迹 +
轮次。你只是依赖补全，而非原生 mapper。这正是 CrewAI 和无框架 agent 达到完整覆盖的方式。

#### 如何解锁每个评估器级别

- **Trace 级别**（Helpfulness、Correctness、Coherence……）需要一个重建出的**轮次**：一个
  用户 prompt + agent 的**最终答案**。两者都要发射（契约第 2–3 条）。Strands 原生发射一个
  `AgentInvocationSpan`；对其他框架 SAES 会从恢复出的 prompt+answer 合成它。
- **Tool 级别**（ToolSelectionAccuracy、ToolParameterAccuracy）需要**工具调用**——名称、
  参数和结果。Strands 发射一个 `ToolExecutionSpan`；其他框架只需在 span 中带上它们的
  Bedrock `toolUse`/`toolResult` Converse 块（SAES 会合成 tool span）。要让
  ToolParameterAccuracy 有意义，参数必须是真实的工具输入。
- **Trajectory 匹配**（确定性）需要有序的工具调用名称，它来自同样的 `toolUse` 块——外加
  ground truth 中的 `expectedTrajectory`。

#### 逐框架说明（来自真实部署）

- **Strands**——无需任何操作。原生 OTEL 发射 `AgentInvocationSpan` +
  `ToolExecutionSpan` + `InferenceSpan`；全部 15 个评估器都可用，包括多轮。这是参考路径。
- **LangGraph**——启用 OpenInference（`LangChainInstrumentor().instrument()`）
  或 OTEL LangChain instrumentor，使 `scope.name` 成为两个 LangChain
  值之一。工具调用流经 Bedrock Converse span → SAES 恢复它们。（在推理配置文件 ARN 上使用
  `ChatBedrockConverse` 时，设置 `provider="anthropic"`。）
- **CrewAI**——它的 scope 是 `openinference.instrumentation.crewai`，原生 mapper
  **不**匹配它，因此原生读取会抛异常——这是预期内的。SAES 从 Converse span 恢复轨迹 +
  答案。当前缺口：CrewAI 每轮的*用户 prompt* 并不总是处于恢复逻辑读取的形态，因此
  ResponseRelevance 可能对它落空（GoalSuccessRate / 工具仍可用）。
- **无框架（裸 boto3）**——无需添加任何埋点：AgentCore 的 botocore Bedrock 埋点已经捕获了
  Converse 请求/响应（包括作为 `body.message` 的最终答案）。SAES 从中重建轮次 +
  工具。只要确保你的 Bedrock 调用走的是被埋点的 client（在 AgentCore Runtime 上默认如此）。

#### 那个习惯

```bash
saes doctor --data-source your_dump.jsonl
```

session id + prompt + completion 上出现 `✓` 表示 session 会重建成功；`✗` 会精确告诉你你的
埋点缺少哪个属性——在信任分数前于源头修复。（§4.0 有一份输出样例。）

---

## 8. 评估场景与结果分析

三组互补的证据，全部使用真实的 Bedrock judge。

### 8.1 每个内置评估器都能区分好与坏

每个内置评估器都被赋予一个专门构造的 GOOD 场景和一个 BAD 场景（手工构造的多轮原生
Session，仅 judge）。一个把所有东西都打高分的评估器毫无用处——这是反向校验，验证它们确实
能区分。

| 评估器 | GOOD | BAD | GOOD 场景 → BAD 场景 |
|---|---|---|---|
| Helpfulness | 0.833 | 0.167 | 清晰的密码重置步骤 → 含糊敷衍 ×2 |
| Correctness | 1.000 | 0.000 | “法国的首都是巴黎” → “……是柏林” |
| Coherence | 1.000 | 0.000 | 一致的周日营业时间 → 先说“每天营业”又说“周日不营业” |
| Conciseness | 1.000 | 0.000 | 对“2+2”回答“4。” → 冗长啰嗦的一大段 |
| Faithfulness | 1.000 | 0.000 | 记得“红色衬衫” → 后来却说“蓝色夹克” |
| Harmfulness | 1.000 | 0.000 | 专业建议 → “散布谣言并搞破坏” |
| InstructionFollowing | 1.000 | 0.000 | 对“只用一个词”的要求回答“蓝色。” → 无视规则的长篇回答 |
| ResponseRelevance | 1.000 | 0.000 | 回答了问题 → 跑题的废话 |
| ContextRelevance | 1.000 | 0.000 | 针对预算问题给出预算摘要 → 却给出饼干食谱 |
| Refusal | 1.000 | 0.000 | 拒绝制造炸弹的请求 → 却提供了操作说明 |
| Stereotyping | 1.000 | 0.000 | 中立描述 → “只有年轻男性才能做到” |
| GoalSuccessRate | 1.000 | 0.000 | 预订航班并确认 → 从未预订 |
| ToolSelectionAccuracy | 1.000 | 0.000 | 天气问题用 `get_weather` → 天气问题却用 `calculate` |
| ToolParameterAccuracy | 1.000 | 0.000 | `get_weather(city=Tokyo)` → 东京问题却用 `get_weather(city=Paris)` |

**14/14 都实现了好 > 坏的区分。** judge 会给出具体理由，例如对于错误事实：*“正确答案是
巴黎，不是柏林。柏林是德国的首都……”*

一个故意做坏的 agent 也被**部署到 AgentCore 上**，并走完整的在线路径打了分（自动 OTEL →
CloudWatch → `saes serve` → judge → 结果）：
Helpfulness **0.0**、InstructionFollowing **0.0**——对比好 agent 在同一路径上的
0.833。区分能力在隔离场景与端到端场景下都成立。

### 8.2 跨框架场景

四个 agent，各用一种不同的框架，全部部署在 AgentCore Runtime 上，全部暴露**相同的两个
工具**并被问**相同的问题**——这样 tool 级别的评估器和 trajectory 匹配器才能同类相比：

```
Tools:  get_weather(city) -> forecast    calculate(expression) -> arithmetic
Prompts: "What's the weather in Tokyo?"       -> get_weather(Tokyo)
         "What is 15% of 240?"                -> calculate(...)
         "Weather in Paris, and what is 12*8?" -> both tools
```

### 8.3 四种框架 × 15 个评估器——矩阵

真实的 AgentCore CloudWatch trace，真实的 Bedrock OpenAI 兼容 judge
（`openai.gpt-oss-20b-1:0`），经由 `saes serve` 的补全后 CloudWatch task。
下面的网格是一次**逐字重跑**（`framework_matrix.py`，保存到
`FRAMEWORK_MATRIX_OUTPUT.txt`）：

| 评估器 | strands | noframe | langgraph | crewai |
|---|---|---|---|---|
| Helpfulness | 0.833 | 0.667 | 0.833 | 0.833 |
| Correctness | 1.000 | 1.000 | 1.000 | 1.000 |
| Coherence | 1.000 | 1.000 | 1.000 | 1.000 |
| Conciseness | 1.000 | 1.000 | 1.000 | 0.500\* |
| Faithfulness | 1.000 | 1.000 | 1.000 | 1.000 |
| Harmfulness | 1.000 | 1.000 | 1.000 | 1.000 |
| InstructionFollowing | 1.000 | 1.000 | 1.000 | 1.000 |
| ResponseRelevance | 1.000 | 1.000 | 1.000 | 1.000 |
| ContextRelevance | 1.000 | 1.000 | 1.000 | 1.000 |
| Refusal | 0.000\*\* | 0.000\*\* | 0.000\*\* | 0.000\*\* |
| Stereotyping | 1.000 | 1.000 | 1.000 | 1.000 |
| GoalSuccessRate | 1.000 | 1.000 | 0.000\* | 0.000\* |
| ToolSelectionAccuracy | (ran†) | 1.000 | 1.000 | 0.000\* |
| ToolParameterAccuracy | (ran†) | 1.000 | 1.000 | 0.000\* |
| TrajectoryAnyOrderMatch | 1.000 | 0.500 | 1.000 | 1.000 |
| **RAN 的评估器数** | **15/15** | **15/15** | **15/15** | **15/15** |

\* 这些是**内容**结果，而非流水线缺口：评估器*运行了*；它打了低分，是因为简短的
ground truth 与更充实的答案不匹配，或该 session 累积了许多漂移的轮次。这张表的重点是*哪些
评估器运行*。个别的内容分数会随 judge 逐次运行而变化（例如某个框架的 Helpfulness 在不同
运行中可能落在 0.667 或 0.833）；而结构——全部 15 个都对全部四种框架运行——是稳定的。
\*\* 在善意流量上 Refusal=0.0 是符合预期的极性：这些 agent 从来没有什么需要拒绝的。
† Strands 的两个 tool 级别单元格在矩阵脚本中显示为空白，仅仅是因为那一趟出现了一次瞬时
judge 错误，且被逐单元格捕获了。Strands **确实**有原生的 `ToolExecutionSpan`（本次已验证：
`session_has_tool_spans=True`），且 CLI 路径（§11 第 3a 步）在全部 10 次工具调用上给它打了
`ToolParameterAccuracy=1.0`——所以这是运行噪声，不是缺口。

### 8.4 分析——每种框架如何达到 15/15

- **Strands**——原生 OTEL 直接发射 `AgentInvocationSpan` + `ToolExecutionSpan`。无需补全。
  参考案例。
- **LangGraph**——原生读取产出 `InferenceSpan`。**轮次补全**合成 agent span；**工具补全**
  恢复工具调用并合成 `ToolExecutionSpan`。
- **无框架 / CrewAI**——原生读取*抛异常*。SAES 替换为一个空 Session，然后：工具补全恢复
  轨迹，role 感知恢复提取最终答案，轮次+tool span 合成构建出提取器所需的原生 span。

**在补全之前，无框架和 CrewAI 只能运行 1/15（trajectory）。补全之后，它们运行 15/15。**
这是框架无关这一主张在完整深度上成立的具体证明——完全在 SAES 采集层实现，不改 agent、不
重新部署（针对同一批已部署 agent 的 CloudWatch 数据验证）。

### 8.5 逐步拆解评估过程

这正是评估某个框架的 agent 时发生的事——§8.3 矩阵每一列背后的具体流水线。无论你是临时运行
（`framework_matrix.py`）、按需运行（`saes run`）还是在线运行（`saes serve`），序列都相同；
只是触发方式不同。**要看部署并评估每种框架的实际可复制粘贴命令，见
[§11 “四框架矩阵”](#11-复现)。** 下面的步骤解释这些命令内部做了什么。

#### 第 0 步——部署 agent（每种框架一次）

每个 agent 暴露相同的两个工具（`get_weather`、`calculate`），并部署到 AgentCore
Runtime。AgentCore 自带 `aws-opentelemetry-distro`，因此 **trace 自动导出到 CloudWatch** 的
`/aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT`——agent 里没有任何遥测代码。每种
框架*发射*什么各不相同，而这个差异就是整个故事：

| 框架 | 埋点来源 | 到达 CloudWatch 的 span |
|---|---|---|
| Strands | 原生 Strands OTEL tracer | `AgentInvocationSpan`、`InferenceSpan`、`ToolExecutionSpan` |
| LangGraph | OpenInference（`LangChainInstrumentor`） | LangChain span → 映射为 `InferenceSpan`（无 tool/agent span） |
| 无框架 | AgentCore 的 **botocore** Bedrock 埋点 | 原始 Converse span：`toolUse`/`toolResult`、带 role 的 `body.message` |
| CrewAI | OpenInference（`.crewai` scope） | 原始 Converse span（scope 对原生 mapper 未知） |

#### 第 1 步——从 CloudWatch 发现 session id

原生 `CloudWatchProvider` 只能*按已知 session id* 读取，所以 SAES 自己负责发现：用一次
Logs Insights 查询在回看窗口内查找去重的 `attributes.session.id`。

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

> 在调用 agent 之后请留出约 90–100 秒：CloudWatch trace 投递 + Logs Insights 索引存在延迟，
> 在线 worker 的轮询间隔会吸收这段延迟。

#### 第 2 步——构建补全后的 task

`build_supplemented_task` 包装原生的 `provider.as_task()`。当对某个 session 调用时，它先做
原生读取，然后按需应用补全。原生读取在不同框架下表现不同——这就是分支点：

```python
from saes.ingest.cloudwatch_task import build_supplemented_task
task = build_supplemented_task(provider, cfg)
out = task(case)              # case.input == session_id
session = out["trajectory"]   # a native Session, ready for the extractor
```

内部逐框架发生了什么（展示真实重建出的 span 类型）：

- **Strands**——原生读取成功，得到
  `['AgentInvocationSpan', 'InferenceSpan', 'ToolExecutionSpan']`。`need_tools`
  和 `need_turn` 都为 false → **不运行任何补全**。Session 原样使用。
- **LangGraph**——原生读取成功，得到 `['AgentInvocationSpan',
  'InferenceSpan']`，但*没有 tool span*。`need_tools` 为 true → SAES 拉取原始 span，恢复
  工具轨迹，并合成 `ToolExecutionSpan`。
- **无框架 / CrewAI**——原生读取**抛出** `SessionNotFoundError`
  （重建出的 span 类型为 `[]`）。SAES 捕获它，替换为空 Session，然后运行完整补全（工具
  轨迹 + role 感知轮次 + tool span）。

原生读取期间 mapper 逐 span 的 WARNING 刷屏被静默为一行 INFO 摘要（F8），例如
`session 60bb9061: recovered 4-step tool trajectory via supplement (native read had failed)`。

#### 第 3 步——重建评估轮次（补全）

对于任何原生读取没有产出所需 span 的框架，SAES 会从原始 CloudWatch 记录重建它们
（`fetch_session_records` → `extract_session_tool_calls`）：

1. **打通 trace_id → session_id。** botocore 工具 span 不携带 `session.id`；
   同一 `trace_id` 中的某个 OpenInference span 携带。SAES 把它们关联起来。
2. **恢复工具轨迹。** 从原始 Converse span 中提取 `toolUse`（名称 + 参数）和
   `toolResult`（内容），与顺序无关。
3. **role 感知地恢复轮次文本。** 从带 role 的 span 中读取用户 prompt 与**最终答案**——
   `body.message`（role=assistant，finish_reason=end_turn）和 `body.{input,output}.messages`。
   （这就是 F11 的修复：最终答案一直都在 CloudWatch 里；SAES 只需从正确的字段读取它，而不
   是靠字符串长度去猜。）
4. **合成原生 span。** 构建一个 `AgentInvocationSpan`（user_prompt +
   agent_response + `available_tools`）以及每次恢复出的工具调用对应一个
   `ToolExecutionSpan`——正是原生 `TraceExtractor` 在 TRACE / SESSION / TOOL 级别所消费的
   确切形态。

在此之后，无论每种框架最初发射的是什么，它们的 Session 都包含相同的可评估结构。

#### 第 4 步——附加 ground truth（只有部分评估器需要）

`Case` 携带可选的 ground truth；每个评估器只读取自己的字段：

```python
from strands_evals import Case
case = Case(name=session_id, input=session_id, session_id=session_id,
            expected_output="The weather in Tokyo is 22C; ...",   # → Correctness
            expected_assertion="Answered weather and math using tools.",  # → GoalSuccessRate
            expected_trajectory=["get_weather", "calculate"])     # → Trajectory*Match
```

那 12 个无需参考答案的 LLM 评估器不需要其中任何东西。

#### 第 5 步——解析评估器并注入 judge

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

`resolve_evaluator` 把 id 映射到原生评估器类，用其 id 给实例命名（使得重复项能在一次
Experiment 中共存），并注入 judge。Trajectory 的 id 则解析为确定性匹配器（无 judge）。

#### 第 6 步——在重建出的 Session 上运行评估器

```python
from strands_evals import Experiment
report = await Experiment(cases=[case], evaluators=[ev]).run_evaluations_async(task)
out = report.detailed_results[0][0]
print(out.score, out.reason)
# 1.0  "The user explicitly asked for the weather in Tokyo. The provided tool, get_weather..."
```

内部，原生 `TraceExtractor` 会按评估器的级别遍历 Session：
- **TRACE 级别**（Helpfulness、Correctness……）——读取 `AgentInvocationSpan` 的
  user_prompt + agent_response。
- **SESSION 级别**（GoalSuccessRate）——读取整段对话。
- **TOOL 级别**（ToolSelectionAccuracy、ToolParameterAccuracy）——读取每个
  `ToolExecutionSpan`（tool_call 名称/参数、tool_result）+ `available_tools`。

由于第 3 步为每种框架都产出了这三类 span，每个级别的评估器都有数据——这就是为什么矩阵全盘
都是 15/15。

#### 第 7 步——聚合、门禁与发射

逐评估器的 `avg / pass% / n`；可选的 CI `gate`（失败时非零退出）；结果写入 JSON/HTML
和/或 CloudWatch EMF + JSON 日志（§9）。在在线 worker 中，这是每个 cycle 的收尾；随后该
session 被标记为已打分，从而永不再被重复打分。

#### 整个过程，浓缩版

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

## 9. 在线/生产环境评估

对线上 agent 的流量进行持续评估。与 `saes eval` 一样，零配置路径**只需 runtime id**——
无需 YAML：

```bash
saes serve myagent-XyZ123                     # continuous; polls every 60s
saes serve myagent-XyZ123 --once              # one cycle (CI/cron)
saes serve myagent-XyZ123 --sampling 5 --session-timeout 30 --all
```

它会推导出 runtime 的 log group，默认使用 12 个无需参考答案的评估器，并**自动创建一个结果
sink**，位于 `/aws/saes/<runtime>-results`（用 `--results-log-group`
覆盖）。评估器标志与 `saes eval` 相同（`-e`、`--all`、`--sampling`、`--judge-model`……），
外加：
- `--session-timeout N`——一个 session 在 N 分钟内没有新 span 后视为“完成”。
- `--interval N`——轮询 cycle 之间的秒数（默认 60）。
- `--state FILE`——跨重启持久化哪些 session 已被打分。
- `--print-scores`——同时在终端打印每个已打分批次的逐评估器分数（无论如何它们都会进
  CloudWatch；在线模式默认将结果写入 CloudWatch，这与打印到终端的 `saes eval` 不同）。

已实测在线：`saes serve saesstrands-... --once` 打分了 3/3 个 session 并把结果写入自动
推导出的 CloudWatch group。

### worker 每个 cycle 做什么

1. **发现**回看窗口内的 session id（Logs Insights）。
2. 通过 span 静默（span-quiescence）**检测完成**——若在 `session.timeout_minutes`
   内没有新 span ⇒ 该 session 完成（与托管 AgentCore 的方式一致）。
3. 依据 `SamplingConfig` **采样**，并带一个滚动窗口的速率上限。
4. 用与 `saes run` 相同的流水线（含补全）**打分**。
5. **发射** EMF 指标 + JSON 结果记录到 CloudWatch。每个 session 最多打分一次（持久化的已
   打分集合）；失败会在下个 cycle 重试。

### 完整配置的替代方案（`--config`）

对于 CI 门禁、自定义 LLM/代码评估器、非 AgentCore 的 log group，或滚动速率上限，请传入一份
YAML，而不是 runtime id：

```yaml
mode: online
dataSource:
  type: cloudwatch
  cloudwatch:
    log_group_names: ["/aws/bedrock-agentcore/runtimes/<agent>-DEFAULT"]
    region: us-east-1
judge: { provider: bedrock, model: "..." }
evaluators: [Builtin.Helpfulness, Builtin.Correctness]
session:  { timeout_minutes: 30 }     # ~ your agent's typical session length
sampling: { percentage: 5.0, max_per_minute: 100 }
resultsSink:
  cloudwatch: { log_group: "/aws/saes/online-results", metrics_namespace: "SAES/Evaluations" }
```

```bash
saes serve -c online.yaml --interval 60 --state state.json   # continuous loop
saes serve -c online.yaml --once                             # one cycle (CI/cron)
```

自定义代码评估器也能作为 Lambda 运行（`online/lambda_evaluator.py`），
而 `cdk/` 会预置 dashboard + 告警 + 最小权限的 worker role。

### 与托管 AgentCore Evaluations 的关系

形态相同（CloudWatch OTEL span → 分组为 session → 采样 → 打分 → EMF
指标 + JSON 日志），但自托管，采用自带的 OpenAI 兼容 judge，并使用
`SAES/Evaluations` 指标命名空间而非托管的那个。

---

## 10. 验证记录：已证实的能力与发现的 bug

下文所有内容都是用**真实 LLM judge** 运行的，而非桩（stub）。正是对真实端到端运行的要求，
才让这些 bug 浮出水面。

### 已端到端证实

- **离线流水线**——真实 Strands agent → OTEL span → 原生 Session → 真实
  Bedrock judge → 带推理的分数（Helpfulness 0.833、Correctness 1.0、
  Coherence 1.0）。
- **在线流水线**——一个真实部署在 **AgentCore Runtime** 上的 agent，自动把 OTEL 导出到
  CloudWatch；`saes serve --once` 发现了静默的 session、给它打分，并把 EMF 指标 + JSON 结果
  写回 CloudWatch。
- **自带 judge**——`openai_compatible` 路径针对 **Bedrock 的 OpenAI API**
  （`openai.gpt-oss-20b-1:0`）验证通过，外加经由同一端点的 **DeepSeek、Kimi 和 Qwen**。
  它们全部合格（支持工具调用）并能端到端打分。
- **judge 可比性的注意事项，已演示：** 在*同一条* trace 上，Qwen 给 Helpfulness 打 1.0，而
  DeepSeek/Kimi 打 0.833——只有在 judge 保持不变时分数才可比。SAES 会在每条结果上标记
  `judgeModel`。
- **框架无关**——Strands、LangGraph、CrewAI 以及一个无框架的 `boto3`
  脚本，全部从真实 AgentCore CloudWatch trace 完成评估；全部达到 15/15 个评估器（§8）。

### 发现并修复的 bug（有信息量的那些）

| # | Bug | 修复 |
|---|---|---|
| F1 | `_final_output(session)` 猜测了不存在的 Session 属性 → 输出为空 | 改为读取 `AgentInvocationSpan.agent_response` |
| F3 | 原生评估器需要结构化输出；只支持文本的端点在运行中途崩溃 | 预检探测（`saes doctor --judge`）在开始前拒绝它们 |
| F7 | `doctor` 字段覆盖在 traceloop/带索引的键上出现假阴性 | 前缀通配别名（`gen_ai.prompt.*`、`traceloop.entity.*`……） |
| F8 | 成功的非 Strands 运行中原生 mapper 的 WARNING 刷屏（约 19 行） | 在读取期间静默那些 logger；发射一行 INFO 摘要 |
| F9 | 原生读取**抛异常**时补全被跳过——恰在最需要时打了空分 | 捕获异常，替换为空 Session，仍然补全 |
| F10 | 非 Strands 的 LLM 评估器返回 None（无 agent span） | 从恢复出的轮次文本合成 `AgentInvocationSpan` |
| F11→ | **错误地**得出结论认为最终答案“不在 CloudWatch 里”、需要在 agent 侧修复 | 其实它*就在那里*（botocore 已捕获）；修复在 SAES 采集层的 **role 感知**提取——把无框架/CrewAI 从 1/15 提升到 13/15 |
| F12 | 最后 2 个 tool 级别的 LLM 评估器对非 Strands 仍无法运行 | 从恢复出的工具调用合成原生 `ToolExecutionSpan` + `available_tools` → 13/15 → 15/15 |
| F13 | 非 Strands 的**多轮** session 错配了轮次（第 3 轮 prompt 配第 1 轮答案）→ 错误的 0.0 分 | 按 `trace_id` 每轮重建一次、按时间排序；每轮一个 `AgentInvocationSpan`（已验证：LangGraph 3 轮 0.0→0.833/1.0……） |
| F14 | 尽管做了静默，mapper 的 WARNING 刷屏仍在多 **session** 评估中泄漏 | 逐 task 的静默器在并发的 `to_thread` task 间竞态；改为引用计数 + 加锁保护 |

> **来自 F9/F11 的教训：** `scored 1/1`（一个 session 被处理了）与非零分数不是一回事，而
> “数据不在那里”是一个应对照原始 span 去验证的断言，而非想当然。记录真实的逐框架数字——
> 并检查真实捕获的 span——正是抓住这两个问题的方式。

### 已知限制

- **Strands scope** span 的本地 `otlp_file` dump 不能往返（内存 mapper 需要
  `ReadableSpan` 对象，而非 dict）。对 Strands 请使用 `live` 内存路径或 CloudWatch 源；
  dict 格式的 CloudWatch/OpenInference/LangChain dump 从文件工作正常。
- 非 Strands agent 合成的 `available_tools` 只携带工具名称（原始 span 缺少工具描述/JSON
  schema）——这是一条保真度说明，而非缺失某个评估器。

---

## 11. 复现

### 单元测试

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]' openai
pytest -q                    # 199 passing
ruff check src/ tests/       # clean
```

### 好/坏区分套件（仅 judge，无需部署）

```bash
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
python builtin_suite.py      # ~26 judge calls; every built-in good>bad
python bad_examples.py       # multi-turn bad sessions score 0.0
```

### 四框架矩阵（在 AgentCore 上部署 + 评估）——具体命令

这是复现 §8.3 的完整序列，附确切命令。agent 源码位于验证工作区
（`/home/ec2-user/saes_run`）的 `agents/{strands,noframework,langgraph,crewai}_tools/`；请按你的
检出目录调整路径。

> **已针对四个已部署 runtime 端到端验证**：矩阵脚本（第 3b 步，打印了完整的 §8.3
> 网格）和逐框架 CLI 路径（第 3a 步，四者都返回 `scored N/N` 并把真实分数写入 CloudWatch——
> Strands/noframe/langgraph 的 tool 级别 1.0，CrewAI trajectory 1.0）。确切输出见下文内嵌。

#### 前置条件

```bash
source .venv/bin/activate    # SAES installed
pip install bedrock-agentcore bedrock-agentcore-starter-toolkit \
            langgraph langchain-aws crewai crewai-tools \
            openinference-instrumentation-langchain openinference-instrumentation-crewai
export BEDROCK_MODEL_ID="us.anthropic.claude-sonnet-4-5-20250929-v1:0"   # the agent's model
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
```

#### 第 1 步——把每个框架的 agent 部署到 AgentCore（每个约 5 分钟，CodeBuild）

每种框架的过程相同；只有目录和名字变化。四者都暴露相同的 `get_weather` + `calculate`
工具：

```bash
cd agents/strands_tools          # then noframework_tools / langgraph_tools / crewai_tools
export AGENTCORE_SUPPRESS_RECOMMENDATION=1
printf '\n\n\n\n\n' | agentcore configure -e agent.py -n saesstrands -rf requirements.txt --create
sed -i 's/ecr_auto_create: false/ecr_auto_create: true/' .bedrock_agentcore.yaml
agentcore deploy -env BEDROCK_MODEL_ID="$BEDROCK_MODEL_ID"
```

坑点（在提供的 agent 源码中已处理）：starter toolkit 可能不会写 `Dockerfile`（每个 agent
都提供了一个）；`ecr_auto_create` 默认为 false（上面的 `sed` 会把它翻过来）；LangGraph 在
推理配置文件 ARN 上使用 `ChatBedrockConverse` 时需要 `provider="anthropic"`。

#### 第 2 步——调用每个 agent（在 CloudWatch 中产生真实的 OTEL trace）

```bash
cd agents/<fw>_tools
agentcore invoke '{"prompt": "What is the weather in Tokyo?"}'
agentcore invoke '{"prompt": "What is 15% of 240?"}'
agentcore invoke '{"prompt": "Weather in Paris, and what is 12*8?"}'
```

评估前等待约 90–100 秒，以完成 trace 投递 + Logs Insights 索引。

#### 第 3a 步——通过 CLI（`saes serve`）评估某一个框架

这是用户实际会输入的内容。为每个框架写一份配置——只有 log group 变化——然后运行一个在线
cycle。**今天已针对全部四个已部署 runtime 验证可用**（每个都返回 `scored N/N`，退出 0）。

先找到 session id（trajectory ground truth 需要它）。session 会随时间移出回看窗口，所以把
`lookback_days` 设为覆盖你的 session——今天已部署的 session 是 44–63 小时前的，因此
`lookback_days: 3`：

```bash
python -c "
from saes.config.schema import CloudWatchSource
from saes.ingest.cloudwatch import build_provider, discover_session_ids
cfg = CloudWatchSource(log_group_names=['/aws/bedrock-agentcore/runtimes/saesnoframe-6AXcAT2oW4-DEFAULT'], region='us-east-1', lookback_days=3)
print(discover_session_ids(build_provider(cfg), cfg))"
# -> ['d8d24446-a5c7-4523-b8f8-dc53a2cfc401', ...]
```

然后写配置 + ground truth 并运行一个 cycle：

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

> 一个有很多工具调用的 session 会在 tool 级别产生很多 judge 调用。LangGraph 的 session
> （13 次工具调用）耗时数分钟——调用 `saes serve` 时请留出充裕的超时。

从 sink 读回分数：

```bash
python -c "
import boto3, json
logs = boto3.client('logs', region_name='us-east-1')
s = logs.describe_log_streams(logGroupName='/aws/saes/fw-results', orderBy='LastEventTime', descending=True, limit=1)['logStreams'][0]['logStreamName']
for e in logs.get_log_events(logGroupName='/aws/saes/fw-results', logStreamName=s, startFromHead=False, limit=50)['events']:
    m = json.loads(e['message'])
    if m.get('type') == 'saes.result': print(m['evaluatorId'], m['score'])"
```

今天读回的真实分数（逐框架，简写）：

```
strands   : Helpfulness 0.833 | ToolParameterAccuracy 1.0 (×10 tool calls) | TrajectoryAnyOrderMatch 1.0
noframe   : Helpfulness 0.833 | ToolSelectionAccuracy 1.0 | ToolParameterAccuracy 1.0 | Trajectory 0.5
langgraph : Helpfulness 0.833 | ToolSelection/ToolParameter across 13 calls | Trajectory 1.0
crewai    : Helpfulness 0.833 | ToolParameterAccuracy (ran; 0.0 on its 8-call session) | Trajectory 1.0
```

通过修改 `log_group_names` 和 session id
（`saesstrands-ZhPiI77pEM-DEFAULT` / `saeslanggraph-vSzHF7G235-DEFAULT`
/ `saescrewai-JjA6Jp5dHw-DEFAULT`）对其他三个框架重复即可。

#### 第 3b 步——四者 × 全部 15 个评估器一张表（矩阵脚本）

要直接复现 §8.3 表格（针对每个内置评估器评估每个已部署框架并打印网格），使用工作区脚本——
它的 judge 通过 `BEDROCK_MODEL_ID` 设置：

```bash
cd /home/ec2-user/saes_run
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"
export BEDROCK_MODEL_ID="openai.gpt-oss-20b-1:0"
python framework_matrix.py       # 4 frameworks × 15 evaluators over real CloudWatch traces (~10 min)
```

今天已运行并打印了完整的 §8.3 网格（保存到
`FRAMEWORK_MATRIX_OUTPUT.txt`）。有两处要匹配你的部署：靠近顶部硬编码的 runtime id
（`AGENTS = {...}`）和 `lookback_days`（当前为 `3`，因为今天的 session 约 2 天前——把它调大，
或重新调用 agent 使其保持新鲜，这样发现才能找到 session）。详尽变体
（`framework_matrix_verbose.py`）还会额外打印重建出的 span 类型 + 每个评估器的 judge 推理。

#### 第 4 步——清理

```bash
for d in strands_tools noframework_tools langgraph_tools crewai_tools; do
  (cd agents/$d && AGENTCORE_SUPPRESS_RECOMMENDATION=1 agentcore destroy)
done
aws logs delete-log-group --log-group-name /aws/saes/fw-results --region us-east-1
```

---

## 12. 故障排查

| 症状 | 原因 / 修复 |
|---|---|
| 运行中途出现 `StructuredOutputException` | judge 端点缺少工具调用。运行 `saes doctor --judge`；换一个合格端点。 |
| `ModuleNotFoundError: openai` | `pip install openai`（`openai_compatible` 需要）。 |
| `saes doctor` 对 session id / prompt 显示 ✗ | 埋点缺少这些 GenAI 属性——请在源头修复。 |
| 本地 dump 得到 0 session / 空分数 | Strands scope 的 dump 不能从文件往返；请用内存或 CloudWatch。dict 格式的 dump 可用。 |
| 某评估器对非 Strands agent 返回空 | 通常数据*确实*在 CloudWatch 中，只是处于不同的 span 形态；补全会处理已知的那些。在断定它缺失之前先检查原始 span。 |
| 分数在多次运行间漂移 | judge 模型变了。保持 judge 不变；SAES 会在每条结果上标记 `judgeModel`。 |
| gate 退出码始终为 0 | 确保配置中有 `gate:` 规则；只有规则失败时退出才非零。 |
