> 英文版见 [README.md](README.md)。

# SAES — Strands Agent Evaluation Suite

面向 AI agent 的开源评估方案，基于 [Strands Agents SDK](https://strandsagents.com/) 构建，并与 Amazon Bedrock AgentCore Observability 集成。

- **自带 judge（Bring your own judge）** —— 任何支持工具调用 / 结构化输出的 OpenAI 兼容端点（OpenAI、Azure、vLLM、LiteLLM、Bedrock）都可用作 LLM-as-a-Judge。运行前可用 `saes doctor --judge` 进行验证。
- **框架无关（Framework-agnostic）** —— 只要 agent 向 CloudWatch 发出 OpenTelemetry GenAI trace，无论使用何种 SDK 或语言均可评估。Strands、LangGraph、CrewAI 以及无框架 agent 都能触达**全部 15 个内置 evaluator**——适配逻辑位于 SAES 的 ingestion 环节，而非你的 agent 中。
- **与 AgentCore 对齐的 evaluator（AgentCore-parity evaluators）** —— 13 个内置项 + 确定性轨迹（trajectory）评分器 + 自定义 LLM/代码 evaluator，全部为原生 `strands-agents-evals`，因此得分与托管的 AgentCore Evaluations 保持一致。

## Quick start

已经在 **AgentCore Runtime** 上部署了 agent？只需一条命令即可评估它——
只要提供 runtime id。无需 YAML、无需 ground truth、无需 trace 管道搭建。要求
**Python 3.12**：

```bash
git clone https://github.com/milan9527/CustomEval.git && cd CustomEval
python3.12 -m venv .venv && source .venv/bin/activate    # activate FIRST
pip install -e '.[dev]' openai aws-bedrock-token-generator

# judge = Amazon Bedrock, via your AWS credentials — no external key
export SAES_JUDGE_API_KEY="$(python -c 'from aws_bedrock_token_generator import provide_token; print(provide_token(region="us-east-1"))')"

saes eval myagent-XXXXXXXXXX --html out/report.html      # ← your AgentCore Runtime id
#   scans the last 7 days by default; add --days 30 for older sessions
#   evaluating /aws/bedrock-agentcore/runtimes/myagent-XXXXXXXXXX-DEFAULT
#     Builtin.Helpfulness        avg=0.833  pass=100%  n=1
#     Builtin.Coherence          avg=1.000  pass=100%  n=1
#     ...
```

`saes eval` 会推导出 runtime 对应的 CloudWatch 日志组，发现其 session，
并用 12 个无需参考答案（reference-free）的内置 evaluator 进行评分。选项与
AgentCore Evaluations 保持一致：

```bash
saes eval --list-evaluators                            # all 13 built-ins + trajectory matchers
saes eval myagent-XXX -e Builtin.Helpfulness,Builtin.Harmfulness   # pick evaluators
saes eval myagent-XXX --all                            # every built-in
saes eval myagent-XXX --sampling 25                    # score 25% of sessions
```

**持续（在线）监控（Continuous (online) monitoring）**同样是这一条命令——
只需把 `eval` 换成 `serve`：

```bash
saes serve myagent-XXX                                 # poll live traffic, score completed sessions
saes serve myagent-XXX --once                          # a single cycle (CI/cron)
```

从头到尾的完整示例（构建 agent → 部署 → 评估）见
**[WALKTHROUGH.zh.md](WALKTHROUGH.zh.md)**。

还没有 agent？1 分钟内即可对随附的 trace 样本评分——参见
[DOCUMENTATION.zh.md §4.0](DOCUMENTATION.zh.md#40-i-just-cloned-this-repo-and-i-have-my-own-agent--where-do-i-start)。

## Documentation

- **[WALKTHROUGH.zh.md](WALKTHROUGH.zh.md)** —— 完整的线性示例：clone →
  构建 agent → 部署到 AgentCore → CloudWatch → 评估。**从这里开始。**
- **[DOCUMENTATION.zh.md](DOCUMENTATION.zh.md)** —— 集中一处的完整参考：
  项目说明、架构、所有使用路径、配置、
  evaluator 目录、各框架支持情况、评估场景 + 结果
  分析、在线评估，以及验证日志。
- **[SPEC.md](SPEC.md)** —— 完整的技术规范。
- **[examples/](examples/)** —— 可部署的 agent 源码，以及结果背后真实运行的**原始记录**：
  框架矩阵、好坏对比的多轮对话、判别（discrimination）套件，以及
  **[真实客户场景](examples/complex_agents/)**（SaaS helpdesk /
  RAG / 预订 / 合规，使用 ground truth + 自定义
  evaluator 进行 on-demand 评估）。

## Status

M1–M3 已完成（离线评估、CloudWatch ingestion、在线 worker、CDK
dashboard）。198 个单元测试。已通过真实 Bedrock judge（离线
+ 在线）、一个真实部署的 AgentCore Runtime agent，以及四种框架（Strands、
LangGraph、CrewAI、无框架）完成端到端验证——全部触达 15/15 个 evaluator。尚未发布。

## License

Apache-2.0
