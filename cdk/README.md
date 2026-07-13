# saes-cdk — CloudWatch dashboard, alarms & worker IAM (T17)

Optional CDK app that provisions observability infra over the
`SAES/Evaluations` EMF namespace the results sink writes to (SPEC §10.2):

- a **dashboard** with per-evaluator score-trend widgets,
- **alarms** that fire when an evaluator's average score drops below a threshold,
- a **least-privilege IAM role** for the online worker (separate from the agent's
  runtime role, mirroring managed AgentCore — SPEC §13).

## Use

```bash
pip install -r requirements.txt        # CDK lib pinned to the CLI's schema
cdk synth  -c agent_id=my-agent -c evaluators="Builtin.Helpfulness,Builtin.Correctness"
cdk deploy -c agent_id=my-agent -c evaluators="Builtin.Helpfulness,Builtin.Correctness" \
           -c score_threshold=0.75
```

Context parameters: `agent_id` (dashboard/dimension key), `evaluators` (comma-
separated ids → one widget + alarm each), `namespace` (default
`SAES/Evaluations`), `score_threshold` (alarm trigger, default 0.75).

> The CDK CLI and `aws-cdk-lib` must share a cloud-assembly schema version. If
> `cdk synth` reports a schema mismatch, either upgrade the CLI
> (`npm i -g aws-cdk`) or adjust the pin in `requirements.txt`.
