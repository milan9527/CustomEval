"""CloudWatch trace source (SPEC §7.2, T8).

The native `strands-agents-evals` `CloudWatchProvider` reads traces **by a known
session_id** (`get_evaluation_data(session_id)`) — verified against v1.0.2, it
has no session-enumeration API. So the one genuinely net-new piece SAES owns
for the CloudWatch path is **session discovery**: a CloudWatch Logs Insights
query that lists distinct `session.id` values in the lookback window. Per-session
reading and mapping are then delegated to the native provider (including its
`as_task()` closure for the runner).

This keeps SAES thin (D1): discovery query + provider construction only; the
provider owns querying, parsing, hierarchy enrichment, and Session mapping.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ..config.schema import CloudWatchSource


def build_provider(cfg: CloudWatchSource) -> Any:
    """Construct a native CloudWatchProvider from a CloudWatchSource config."""
    from strands_evals.providers import CloudWatchProvider

    kwargs: dict[str, Any] = {"lookback_days": cfg.lookback_days}
    if cfg.region:
        kwargs["region"] = cfg.region
    if cfg.agent_name:
        kwargs["agent_name"] = cfg.agent_name
    if cfg.log_group_names:
        # native provider takes a single log_group; use the first and note extras.
        kwargs["log_group"] = cfg.log_group_names[0]
    return CloudWatchProvider(**kwargs)


def discover_session_ids(provider: Any, cfg: CloudWatchSource) -> list[str]:
    """List distinct session ids in the lookback window via Logs Insights.

    Reuses the provider's configured boto3 client and log group. SAES owns this
    because the native provider only reads by an already-known session id.
    """
    client = _provider_client(provider)
    log_group = _provider_log_group(provider)
    if client is None or not log_group:
        raise RuntimeError(
            "cannot discover sessions: native CloudWatchProvider did not expose a "
            "client/log group. Provide explicit session ids instead."
        )

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=cfg.lookback_days)
    query = (
        "fields attributes.session.id as sid "
        "| filter ispresent(sid) "
        "| stats count(*) by sid "
        "| limit 1000"
    )

    resp = client.start_query(
        logGroupName=log_group,
        startTime=int(start.timestamp()),
        endTime=int(end.timestamp()),
        queryString=query,
    )
    results = _poll(client, resp["queryId"])
    return _extract_session_ids(results)


def discover_sessions_with_last_seen(
    provider: Any, cfg: CloudWatchSource
) -> list[tuple[str, int]]:
    """List (session_id, last_span_epoch_ms) in the lookback window.

    Used by the online worker's session-completion detection (T14): a session
    is complete once `now - last_span_ms >= session.timeout`. Uses the standard
    Logs Insights `@timestamp` field aggregated per session.
    """
    client = _provider_client(provider)
    log_group = _provider_log_group(provider)
    if client is None or not log_group:
        raise RuntimeError(
            "cannot discover sessions: native CloudWatchProvider did not expose a "
            "client/log group."
        )

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=cfg.lookback_days)
    # @timestamp is CloudWatch Logs Insights' built-in per-event time (ms since
    # epoch); max() gives each session's most recent span.
    query = (
        "fields attributes.session.id as sid, @timestamp as ts "
        "| filter ispresent(sid) "
        "| stats max(ts) as last_ts by sid "
        "| limit 1000"
    )
    resp = client.start_query(
        logGroupName=log_group,
        startTime=int(start.timestamp()),
        endTime=int(end.timestamp()),
        queryString=query,
    )
    results = _poll(client, resp["queryId"])
    return _extract_session_last_seen(results)


def fetch_session_records(
    provider: Any, cfg: CloudWatchSource, session_id: str
) -> list[dict[str, Any]]:
    """Fetch raw span records for a session, INCLUDING tool spans that carry no
    session.id (bridged via shared trace_id).

    (1) Fetch every record that carries this `session.id` directly — this alone is
    the whole session for agents whose spans all tag session.id (Strands, and the
    Claude Agent SDK, whose log events are keyed by `prompt.id` and carry NO
    trace_id). (2) Additionally bridge via shared `trace_id`: fetch all records in
    the session's traces, to catch session-id-less tool spans (the botocore/
    LangGraph case). Union the two. Feeds
    `tool_supplement.extract_session_tool_calls` (SPEC F6 fix)."""
    import json

    client = _provider_client(provider)
    log_group = _provider_log_group(provider)
    if client is None or not log_group:
        return []

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=cfg.lookback_days)

    def _run(query: str) -> list[dict[str, Any]]:
        resp = client.start_query(
            logGroupName=log_group,
            startTime=int(start.timestamp()),
            endTime=int(end.timestamp()),
            queryString=query,
        )
        out: list[dict[str, Any]] = []
        for row in _poll(client, resp["queryId"]):
            msg = next((f["value"] for f in row if f.get("field") == "@message"), None)
            if not msg:
                continue
            try:
                out.append(json.loads(msg))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    # step 1: every record carrying this session.id directly
    session_recs = _run(
        f"fields @message | filter attributes.session.id = '{session_id}' | limit 2000"
    )

    # step 2: bridge via trace_id to catch session-id-less tool spans. Agents whose
    # spans have no traceId (e.g. the Claude Agent SDK) skip this — step 1 is whole.
    trace_ids: set[str] = set()
    for obj in session_recs:
        tid = obj.get("traceId") or obj.get("trace_id")
        if tid:
            trace_ids.add(str(tid))
    trace_recs: list[dict[str, Any]] = []
    if trace_ids:
        filt = " or ".join(f"traceId = '{t}'" for t in trace_ids)
        trace_recs = _run(f"fields @message | filter {filt} | limit 2000")

    # union, de-duplicated (a record may match both queries)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for obj in [*session_recs, *trace_recs]:
        key = json.dumps(obj, sort_keys=True)
        if key not in seen:
            seen.add(key)
            records.append(obj)
    return records


def _extract_session_ids(results: list[list[dict[str, Any]]]) -> list[str]:
    ids: list[str] = []
    for row in results:
        for field in row:
            if field.get("field") == "sid" and field.get("value"):
                ids.append(field["value"])
    # de-dup, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _extract_session_last_seen(
    results: list[list[dict[str, Any]]],
) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for row in results:
        fields = {f.get("field"): f.get("value") for f in row}
        sid = fields.get("sid")
        last = fields.get("last_ts")
        if sid and last is not None:
            out.append((sid, _to_ms(last)))
    return out


def _to_ms(value: Any) -> int:
    """Parse a Logs Insights timestamp value to epoch ms.

    Logs Insights returns @timestamp as a string like '2026-07-11 12:00:00.000'
    (UTC) or, for max(@timestamp), a numeric ms value depending on the field.
    Handle both.
    """
    s = str(value).strip()
    try:
        return int(float(s))
    except ValueError:
        pass
    # ISO-ish 'YYYY-MM-DD HH:MM:SS.mmm' in UTC
    from datetime import datetime as _dt

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = _dt.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"cannot parse Logs Insights timestamp: {value!r}")


def _poll(client: Any, query_id: str, max_polls: int = 60) -> list[list[dict[str, Any]]]:
    import time

    for _ in range(max_polls):
        resp = client.get_query_results(queryId=query_id)
        status = resp.get("status")
        if status == "Complete":
            return resp.get("results", [])
        if status in ("Failed", "Cancelled", "Timeout"):
            raise RuntimeError(f"session-discovery query {status.lower()}")
        time.sleep(1)
    raise RuntimeError("session-discovery query did not complete in time")


def _provider_client(provider: Any) -> Any:
    return getattr(provider, "_client", None) or getattr(provider, "client", None)


def _provider_log_group(provider: Any) -> str | None:
    return getattr(provider, "_log_group", None) or getattr(provider, "log_group", None)


__all__ = [
    "build_provider",
    "discover_session_ids",
    "discover_sessions_with_last_seen",
    "fetch_session_records",
]
