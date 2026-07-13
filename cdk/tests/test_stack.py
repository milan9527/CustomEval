"""T17 — CDK stack synth assertions (no deploy).

Run from the cdk/ directory: `pytest tests/` (needs aws-cdk-lib + this package
importable). Kept separate from the main src/ test suite because it depends on
the CDK toolchain.
"""

import aws_cdk as cdk
from aws_cdk.assertions import Match, Template

from saes_cdk.stack import SaesObservabilityStack


def _template(evaluators):
    app = cdk.App()
    stack = SaesObservabilityStack(
        app, "Test", evaluator_ids=evaluators, agent_id="my-agent"
    )
    return Template.from_stack(stack)


def test_one_alarm_and_widget_per_evaluator():
    t = _template(["Builtin.Helpfulness", "Builtin.Correctness", "Builtin.Coherence"])
    t.resource_count_is("AWS::CloudWatch::Alarm", 3)
    t.resource_count_is("AWS::CloudWatch::Dashboard", 1)


def test_alarm_threshold_and_comparison():
    t = _template(["Builtin.Helpfulness"])
    t.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "Threshold": 0.75,
            "ComparisonOperator": "LessThanThreshold",
            "Namespace": "SAES/Evaluations",
            "MetricName": "Score",
        },
    )


def test_worker_role_least_privilege():
    t = _template(["Builtin.Helpfulness"])
    t.resource_count_is("AWS::IAM::Role", 1)
    # PutMetricData is namespace-scoped
    t.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "Action": "cloudwatch:PutMetricData",
                                "Condition": {
                                    "StringEquals": {"cloudwatch:namespace": "SAES/Evaluations"}
                                },
                            }
                        )
                    ]
                )
            }
        },
    )


def test_custom_threshold_and_namespace():
    app = cdk.App()
    stack = SaesObservabilityStack(
        app,
        "T2",
        evaluator_ids=["Builtin.Helpfulness"],
        agent_id="a",
        namespace="Custom/NS",
        score_alarm_threshold=0.9,
    )
    t = Template.from_stack(stack)
    t.has_resource_properties(
        "AWS::CloudWatch::Alarm", {"Threshold": 0.9, "Namespace": "Custom/NS"}
    )
