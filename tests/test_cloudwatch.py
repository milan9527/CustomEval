"""T8/T9 CloudWatch source + results sink tests (stubbed boto3, no AWS)."""

import json

from saes.config.schema import CloudWatchSink, CloudWatchSource
from saes.ingest.cloudwatch import discover_session_ids
from saes.report import build_report, emit_to_cloudwatch
from saes.report.build import ReportDocument, ResultRow

# ---- T8: session discovery --------------------------------------------------

class _StubLogsClient:
    def __init__(self, results):
        self._results = results
        self.queries = []

    def start_query(self, **kw):
        self.queries.append(kw)
        return {"queryId": "q1"}

    def get_query_results(self, queryId):
        return {"status": "Complete", "results": self._results}


class _StubProvider:
    def __init__(self, client):
        self._client = client
        self._log_group = "/aws/bedrock-agentcore/my-agent"


def test_discover_session_ids_dedups_and_orders():
    results = [
        [{"field": "sid", "value": "s1"}, {"field": "count(*)", "value": "3"}],
        [{"field": "sid", "value": "s2"}, {"field": "count(*)", "value": "1"}],
        [{"field": "sid", "value": "s1"}, {"field": "count(*)", "value": "2"}],
    ]
    provider = _StubProvider(_StubLogsClient(results))
    cfg = CloudWatchSource(log_group_names=["/aws/x"], region="us-east-1")
    ids = discover_session_ids(provider, cfg)
    assert ids == ["s1", "s2"]


def test_discover_uses_lookback_window():
    client = _StubLogsClient([[{"field": "sid", "value": "s1"}]])
    provider = _StubProvider(client)
    cfg = CloudWatchSource(log_group_names=["/aws/x"], lookback_days=7)
    discover_session_ids(provider, cfg)
    q = client.queries[0]
    assert q["logGroupName"] == "/aws/bedrock-agentcore/my-agent"
    assert q["endTime"] - q["startTime"] == 7 * 86400


def test_discover_sessions_with_last_seen_numeric_ms():
    from saes.ingest.cloudwatch import discover_sessions_with_last_seen

    results = [
        [{"field": "sid", "value": "s1"}, {"field": "last_ts", "value": "1700000000000"}],
        [{"field": "sid", "value": "s2"}, {"field": "last_ts", "value": "1700000060000"}],
    ]
    provider = _StubProvider(_StubLogsClient(results))
    out = discover_sessions_with_last_seen(provider, CloudWatchSource(log_group_names=["/x"]))
    assert out == [("s1", 1700000000000), ("s2", 1700000060000)]


def test_discover_sessions_with_last_seen_iso_timestamp():
    from saes.ingest.cloudwatch import discover_sessions_with_last_seen

    results = [
        [{"field": "sid", "value": "s1"}, {"field": "last_ts", "value": "2026-07-11 12:00:00.000"}]
    ]
    provider = _StubProvider(_StubLogsClient(results))
    out = discover_sessions_with_last_seen(provider, CloudWatchSource(log_group_names=["/x"]))
    sid, ms = out[0]
    assert sid == "s1"
    # 2026-07-11 12:00:00 UTC in ms
    assert ms == 1783771200000


# ---- T9: CloudWatch results sink --------------------------------------------

def _doc():
    return ReportDocument(
        config_name="demo",
        judge_model="gpt-4.1",
        evaluator_ids=["Builtin.Helpfulness"],
        session_ids=["s1"],
        overall_score=0.8,
        aggregates={
            "Builtin.Helpfulness": {"avg": 0.8, "pass_rate": 1.0, "n": 1.0, "errored": 0.0}
        },
        rows=[
            ResultRow(
                evaluator_id="Builtin.Helpfulness",
                session_id="s1",
                score=0.8,
                reason="advanced the user",
                label="Very Helpful",
            )
        ],
    )


class _CapturingLogs:
    def __init__(self):
        self.events = []
        self.groups = []
        self.streams = []

    def create_log_group(self, logGroupName):
        self.groups.append(logGroupName)

    def create_log_stream(self, logGroupName, logStreamName):
        self.streams.append((logGroupName, logStreamName))

    def put_log_events(self, logGroupName, logStreamName, logEvents):
        self.events.extend(logEvents)


def test_sink_noop_when_unconfigured():
    assert emit_to_cloudwatch(_doc(), None) is False


def test_sink_emits_emf_and_records():
    logs = _CapturingLogs()
    sink = CloudWatchSink(
        log_group="/aws/saes/evaluations",
        metrics_namespace="SAES/Evaluations",
        dimensions=["evaluatorId", "agentId"],
    )
    emitted = emit_to_cloudwatch(_doc(), sink, client=logs, now_ms=1700000000000)
    assert emitted is True

    messages = [json.loads(e["message"]) for e in logs.events]
    emf = [m for m in messages if "_aws" in m]
    records = [m for m in messages if m.get("type") == "saes.result"]

    # one EMF metric event for the evaluator
    assert len(emf) == 1
    metric = emf[0]
    assert metric["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "SAES/Evaluations"
    assert metric["Score"] == 0.8
    assert metric["PassRate"] == 1.0
    assert metric["evaluatorId"] == "Builtin.Helpfulness"
    assert metric["_aws"]["Timestamp"] == 1700000000000

    # one result record carrying judge reasoning
    assert len(records) == 1
    assert records[0]["reason"] == "advanced the user"
    assert records[0]["sessionId"] == "s1"


def test_sink_creates_group_and_stream():
    logs = _CapturingLogs()
    sink = CloudWatchSink(log_group="/aws/saes/evaluations")
    emit_to_cloudwatch(_doc(), sink, client=logs, now_ms=1)
    assert "/aws/saes/evaluations" in logs.groups
    assert logs.streams[0][0] == "/aws/saes/evaluations"


def test_build_report_then_emit_integration():
    from saes.run.runner import RunResult

    class _R:
        detailed_results = [
            [type("O", (), {"score": 0.8, "reason": "ok", "label": "L", "test_pass": True})()]
        ]
        cases = [{"evaluator": "Builtin.Helpfulness", "name": "s1"}]
        overall_score = 0.8

    rr = RunResult(
        config_name="demo",
        judge_model="m",
        report=_R(),
        evaluator_ids=["Builtin.Helpfulness"],
        session_ids=["s1"],
        aggregates={
            "Builtin.Helpfulness": {"avg": 0.8, "pass_rate": 1.0, "n": 1.0, "errored": 0.0}
        },
    )
    logs = _CapturingLogs()
    doc = build_report(rr)
    assert emit_to_cloudwatch(doc, CloudWatchSink(log_group="/g"), client=logs, now_ms=1) is True
    assert len(logs.events) == 2  # 1 EMF + 1 record
