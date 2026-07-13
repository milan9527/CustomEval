#!/usr/bin/env python3
"""CDK app entrypoint for SAES observability (T17).

Synth/deploy the dashboard + alarms + worker IAM role. Configuration comes from
context so the same app serves any agent:

    cdk synth -c agent_id=my-agent \
              -c evaluators=Builtin.Helpfulness,Builtin.Correctness

    cdk deploy -c agent_id=my-agent -c evaluators=...
"""

import aws_cdk as cdk

from saes_cdk.stack import DEFAULT_NAMESPACE, SaesObservabilityStack

app = cdk.App()

agent_id = app.node.try_get_context("agent_id") or "my-agent"
evaluators_ctx = app.node.try_get_context("evaluators") or "Builtin.Helpfulness,Builtin.Correctness"
evaluator_ids = [e.strip() for e in evaluators_ctx.split(",") if e.strip()]
namespace = app.node.try_get_context("namespace") or DEFAULT_NAMESPACE
threshold = float(app.node.try_get_context("score_threshold") or 0.75)

SaesObservabilityStack(
    app,
    f"Saes-{agent_id}",
    evaluator_ids=evaluator_ids,
    agent_id=agent_id,
    namespace=namespace,
    score_alarm_threshold=threshold,
)

app.synth()
