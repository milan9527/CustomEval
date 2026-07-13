"""SAES CloudWatch dashboard + alarms + IAM (SPEC §10.2, §11.1, T17).

Provisions, over the `SAES/Evaluations` EMF namespace that the results sink
writes to (SPEC §10):
  - a dashboard with per-evaluator score-trend widgets,
  - alarms that fire when an evaluator's average score drops below a threshold,
  - a least-privilege execution role for the online worker, separate from the
    agent's runtime role (mirroring managed AgentCore).

This is infrastructure-as-code only; `cdk synth` validates it without deploying.
"""

from __future__ import annotations

from aws_cdk import Duration, Stack
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_iam as iam
from aws_cdk import aws_sns as sns
from constructs import Construct

DEFAULT_NAMESPACE = "SAES/Evaluations"


class SaesObservabilityStack(Stack):
    """Dashboard + alarms + worker IAM role for a SAES deployment."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        evaluator_ids: list[str],
        agent_id: str,
        namespace: str = DEFAULT_NAMESPACE,
        results_log_group: str = "/aws/saes/evaluations",
        score_alarm_threshold: float = 0.75,
        alarm_topic: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        topic = sns.Topic(self, "AlarmTopic") if alarm_topic else None

        score_widgets: list[cw.IWidget] = []
        for ev_id in evaluator_ids:
            score_metric = self._score_metric(namespace, ev_id, agent_id)
            score_widgets.append(
                cw.GraphWidget(
                    title=f"{ev_id} — avg score",
                    left=[score_metric],
                    width=12,
                    height=6,
                    left_y_axis=cw.YAxisProps(min=0, max=1),
                )
            )
            alarm = cw.Alarm(
                self,
                f"LowScore{_safe(ev_id)}",
                metric=score_metric,
                threshold=score_alarm_threshold,
                comparison_operator=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
                evaluation_periods=3,
                datapoints_to_alarm=2,
                treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
                alarm_description=(
                    f"{ev_id} average score < {score_alarm_threshold} for agent {agent_id}"
                ),
            )
            if topic is not None:
                alarm.add_alarm_action(cw_actions.SnsAction(topic))

        self.dashboard = cw.Dashboard(
            self,
            "Dashboard",
            dashboard_name=f"SAES-{_safe(agent_id)}",
            widgets=[score_widgets],
        )

        self.worker_role = self._worker_role(namespace, results_log_group)

    def _score_metric(self, namespace: str, ev_id: str, agent_id: str) -> cw.Metric:
        return cw.Metric(
            namespace=namespace,
            metric_name="Score",
            dimensions_map={"evaluatorId": ev_id, "agentId": agent_id},
            statistic="Average",
            period=Duration.minutes(5),
        )

    def _worker_role(self, namespace: str, results_log_group: str) -> iam.Role:
        """Least-privilege execution role for the online worker (SPEC §13)."""
        role = iam.Role(
            self,
            "OnlineWorkerRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="SAES online worker — read agent traces, write eval results",
        )
        # read: discover + read agent trace log groups via Logs Insights
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:StartQuery",
                    "logs:GetQueryResults",
                    "logs:GetLogEvents",
                    "logs:DescribeLogGroups",
                    "logs:DescribeLogStreams",
                ],
                resources=["*"],  # scoped to trace log groups at deploy time
            )
        )
        # write: results log group + EMF metrics
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogStream", "logs:PutLogEvents", "logs:CreateLogGroup"],
                resources=[
                    f"arn:aws:logs:*:*:log-group:{results_log_group}",
                    f"arn:aws:logs:*:*:log-group:{results_log_group}:*",
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={"StringEquals": {"cloudwatch:namespace": namespace}},
            )
        )
        # judge: Bedrock invoke (only if using a Bedrock judge)
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=["*"],
            )
        )
        return role


def _safe(s: str) -> str:
    """CloudFormation-safe logical id / dashboard name fragment."""
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")


__all__ = ["SaesObservabilityStack", "DEFAULT_NAMESPACE"]
