"""CloudWatch results sink (SPEC §10, T9).

Emits evaluation results to CloudWatch so quality scores sit alongside
operational metrics in the AgentCore gen-AI observability views:

- **Metrics (EMF):** per-evaluator avg / pass_rate / errored as custom metrics
  under `metrics_namespace`, dimensioned by agent + evaluator. EMF is used
  (not put_metric_data) so metrics AND the structured record land in one
  `put_log_events` call and render in the gen-AI observability dashboard.
- **Logs:** full per-row result records (incl. judge reasoning) as JSON to a
  results log group for drill-down.

Uses the ambient boto3 session (IAM role) — never inline credentials. No-ops
cleanly when `resultsSink.cloudwatch` is absent.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..config.schema import CloudWatchSink
from .build import ReportDocument


def emit_to_cloudwatch(
    doc: ReportDocument,
    sink: CloudWatchSink | None,
    *,
    region: str | None = None,
    client: Any | None = None,
    now_ms: int | None = None,
) -> bool:
    """Emit `doc` to CloudWatch. Returns True if emitted, False if no sink.

    `client`/`now_ms` are injectable for testing (stub logs client, fixed time).
    """
    if sink is None:
        return False

    logs = client or _logs_client(region)
    ts = now_ms if now_ms is not None else int(time.time() * 1000)

    _ensure_log_group(logs, sink.log_group)
    events = _build_log_events(doc, sink, ts)
    if events:
        logs.put_log_events(
            logGroupName=sink.log_group,
            logStreamName=_ensure_log_stream(logs, sink.log_group, doc.config_name),
            logEvents=events,
        )
    return True


def _build_log_events(
    doc: ReportDocument, sink: CloudWatchSink, ts: int
) -> list[dict[str, Any]]:
    """One EMF metric event per evaluator (avg/pass_rate/errored) plus one JSON
    record event per result row (judge reasoning drill-down)."""
    events: list[dict[str, Any]] = []

    for ev_id, stats in doc.aggregates.items():
        events.append(
            {"timestamp": ts, "message": json.dumps(_emf_metric(doc, sink, ev_id, stats, ts))}
        )

    for row in doc.rows:
        events.append(
            {
                "timestamp": ts,
                "message": json.dumps(
                    {
                        "type": "saes.result",
                        "config": doc.config_name,
                        "judgeModel": doc.judge_model,
                        "evaluatorId": row.evaluator_id,
                        "sessionId": row.session_id,
                        "score": row.score,
                        "label": row.label,
                        "reason": row.reason,
                        "errored": row.errored,
                    }
                ),
            }
        )
    return events


def _emf_metric(
    doc: ReportDocument, sink: CloudWatchSink, ev_id: str, stats: dict[str, float], ts: int
) -> dict[str, Any]:
    """CloudWatch Embedded Metric Format payload for one evaluator."""
    dims = list(sink.dimensions) or ["evaluatorId"]
    dim_values = {
        "evaluatorId": ev_id,
        "agentId": doc.config_name,
        "env": "default",
    }
    return {
        "_aws": {
            "Timestamp": ts,
            "CloudWatchMetrics": [
                {
                    "Namespace": sink.metrics_namespace,
                    "Dimensions": [dims],
                    "Metrics": [
                        {"Name": "Score", "Unit": "None"},
                        {"Name": "PassRate", "Unit": "None"},
                        {"Name": "Errored", "Unit": "Count"},
                    ],
                }
            ],
        },
        **{d: dim_values.get(d, "unknown") for d in dims},
        "evaluatorId": ev_id,
        "Score": stats.get("avg", 0.0),
        "PassRate": stats.get("pass_rate", 0.0),
        "Errored": stats.get("errored", 0.0),
    }


def _logs_client(region: str | None) -> Any:
    import boto3

    return boto3.client("logs", region_name=region) if region else boto3.client("logs")


def _ensure_log_group(logs: Any, name: str) -> None:
    try:
        logs.create_log_group(logGroupName=name)
    except Exception:  # noqa: BLE001 - already-exists is the common case
        pass


def _ensure_log_stream(logs: Any, group: str, config_name: str) -> str:
    stream = f"saes/{config_name}"
    try:
        logs.create_log_stream(logGroupName=group, logStreamName=stream)
    except Exception:  # noqa: BLE001 - already-exists
        pass
    return stream


__all__ = ["emit_to_cloudwatch"]
