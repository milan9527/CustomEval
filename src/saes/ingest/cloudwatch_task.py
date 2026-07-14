"""CloudWatch evaluation task with tool-trajectory supplement (F6 fix).

Shared by BOTH the on-demand runner and the online worker so the supplement
(recovering non-Strands tool trajectories the native mapper misses) applies
uniformly. Wraps the native `provider.as_task()`: after the native Session is
built, if it has no tool spans, fetch the raw CloudWatch records for that
session, extract the tool trajectory (`tool_supplement`), and attach it to the
Session as `_saes_tool_names` — which `TrajectoryMatchEvaluator` uses as a
fallback. Best-effort: never raises into the run.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from ..config.schema import CloudWatchSource
from .cloudwatch import fetch_session_records
from .tool_supplement import extract_session_tool_calls

_log = logging.getLogger("saes.ingest.cloudwatch_task")

# Native mapper loggers that emit a per-span WARNING for every span they can't
# convert (e.g. "Missing required fields for tool span", "No agent_response ...").
# For non-Strands agents these fire dozens of times on an otherwise-successful
# run (SAES's supplement recovers the trajectory), so they read as spurious
# errors. We quiet them to ERROR during the native read and summarize instead.
_NATIVE_MAPPER_LOGGERS = (
    "strands_evals.mappers.openinference_session_mapper",
    "strands_evals.mappers.cloudwatch_session_mapper",
    "strands_evals.mappers.langchain_otel_session_mapper",
    "strands_evals.mappers.strands_in_memory_session_mapper",
)


@contextlib.contextmanager
def _quiet_native_mapper_warnings():
    """Temporarily suppress the native mappers' per-span WARNING spam."""
    saved = {}
    for name in _NATIVE_MAPPER_LOGGERS:
        lg = logging.getLogger(name)
        saved[name] = lg.level
        lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)


def session_has_tool_spans(session: Any) -> bool:
    for trace in getattr(session, "traces", []) or []:
        for span in getattr(trace, "spans", []) or []:
            if type(span).__name__ == "ToolExecutionSpan" or getattr(span, "tool_call", None):
                return True
    return False


def _extract_for_session(provider: Any, cw_cfg: CloudWatchSource, session_id: str):
    """Run the raw-span supplement for one session; return the matching
    SessionToolCalls (trajectory + recovered user/assistant text) or None."""
    try:
        sessions = extract_session_tool_calls(fetch_session_records(provider, cw_cfg, session_id))
    except Exception:  # noqa: BLE001 - best-effort; never break a run
        return None
    match = next((s for s in sessions if s.session_id == session_id), None)
    if match is None:
        # fall back to the first session that has any signal
        match = next(
            (s for s in sessions if s.trajectory or s.user_prompts or s.assistant_texts),
            None,
        )
    return match


def supplement_trajectory(provider: Any, cw_cfg: CloudWatchSource, session_id: str) -> list[str]:
    """Extract a non-Strands tool trajectory from raw CloudWatch spans."""
    s = _extract_for_session(provider, cw_cfg, session_id)
    return s.trajectory if s else []


def _msg_text(message: Any) -> str:
    """Extract plain text from a native message's content blocks."""
    parts = []
    for c in getattr(message, "content", None) or []:
        t = getattr(c, "text", None)
        if t:
            parts.append(t)
    return " ".join(parts).strip()


def session_has_agent_span(session: Any) -> bool:
    for trace in getattr(session, "traces", []) or []:
        for span in getattr(trace, "spans", []) or []:
            if type(span).__name__ == "AgentInvocationSpan":
                return True
    return False


def _synth_tool_spans(tool_calls: Any, sid: str, now: Any) -> list[Any]:
    """Build native ToolExecutionSpans from recovered ToolCallRecords so the two
    tool-level LLM evaluators (ToolSelectionAccuracy/ToolParameterAccuracy) can
    run for non-Strands agents. Each record carries name/arguments/result — the
    exact fields ToolCall/ToolResult need. Best-effort per record."""
    spans: list[Any] = []
    try:
        from strands_evals.types.trace import (
            SpanInfo,
            ToolCall,
            ToolExecutionSpan,
            ToolResult,
        )
    except Exception:  # noqa: BLE001
        return spans
    for i, rec in enumerate(tool_calls or []):
        name = getattr(rec, "name", None)
        if not name:
            continue
        cid = getattr(rec, "tool_call_id", None) or f"saes-tool-{i}"
        try:
            spans.append(ToolExecutionSpan(
                span_info=SpanInfo(trace_id="saes-synth", span_id=f"saes-tool-{i}",
                                   session_id=sid, start_time=now, end_time=now),
                tool_call=ToolCall(name=name,
                                   arguments=getattr(rec, "arguments", None) or {},
                                   tool_call_id=cid),
                tool_result=ToolResult(content=str(getattr(rec, "result", "") or ""),
                                       tool_call_id=cid),
            ))
        except Exception:  # noqa: BLE001 - skip a malformed record, keep the rest
            continue
    return spans


def _tool_configs(tool_spans: list[Any]) -> list[Any]:
    """Distinct-by-name ToolConfigs from synthesized tool spans (names only —
    raw spans carry no tool descriptions/schemas)."""
    from strands_evals.types.trace import ToolConfig

    out, seen = [], set()
    for ts in tool_spans:
        n = ts.tool_call.name
        if n not in seen:
            seen.add(n)
            try:
                out.append(ToolConfig(name=n))
            except Exception:  # noqa: BLE001
                pass
    return out


def supplement_turns(session: Any, user_prompt: str = "", agent_response: str = "",
                     tool_calls: Any = None, turns: Any = None) -> bool:
    """Synthesize AgentInvocationSpan(s) (+ matching ToolExecutionSpans) so
    TRACE/SESSION/TOOL-level evaluators can run when the native mapper produced
    none (common for non-Strands agents).

    Preferred path: pass `turns` (a list of reconstructed Turn objects, one per
    AgentCore trace). Each becomes its own Trace with an AgentInvocationSpan
    pairing that turn's prompt + answer + tools — a faithful **multi-turn**
    Session. Falls back to a single `user_prompt`/`agent_response`(+`tool_calls`)
    turn, else lifts text from the session's own InferenceSpan messages. Works on
    an empty Session (creates traces). Returns True if anything was synthesized.
    Best-effort; never raises.
    """
    try:
        from datetime import datetime, timezone

        from strands_evals.types.trace import AgentInvocationSpan, SpanInfo, Trace

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        sid = getattr(session, "session_id", "s")

        # --- build the list of (turn) span-groups to add ---------------------
        new_traces: list[Any] = []

        def _make_turn_trace(idx: int, up: str, ar: str, tcs: Any) -> Any:
            tool_spans = _synth_tool_spans(tcs, sid, base)
            agent_span = AgentInvocationSpan(
                span_info=SpanInfo(trace_id=f"saes-synth-{idx}", span_id=f"saes-synth-{idx}",
                                   session_id=sid, start_time=base, end_time=base),
                user_prompt=up or "(unknown)", agent_response=ar or "(unknown)",
                available_tools=_tool_configs(tool_spans), system_prompt="",
            )
            return Trace(spans=[agent_span, *tool_spans],
                         trace_id=f"saes-synth-{idx}", session_id=sid)

        if turns:
            for i, t in enumerate(turns):
                up = getattr(t, "user_prompt", "") or ""
                ar = getattr(t, "agent_response", "") or ""
                tcs = getattr(t, "tool_calls", None)
                if up or ar or tcs:
                    new_traces.append(_make_turn_trace(i, up, ar, tcs))
        else:
            # single-turn fallback: explicit args, else lift from InferenceSpans
            if not (user_prompt or agent_response):
                for trace in getattr(session, "traces", None) or []:
                    for span in getattr(trace, "spans", []) or []:
                        for m in getattr(span, "messages", None) or []:
                            role = str(getattr(m, "role", "")).lower()
                            txt = _msg_text(m)
                            if "user" in role and not user_prompt and txt:
                                user_prompt = txt
                            if "assistant" in role and txt:
                                agent_response = txt
            if not (user_prompt or agent_response):
                return False
            new_traces.append(_make_turn_trace(0, user_prompt, agent_response, tool_calls))

        if not new_traces:
            return False

        existing = getattr(session, "traces", None)
        if existing:
            # append synthesized turns' spans into the first existing trace so the
            # extractor sees them (single-trace sessions), but keep multi-turn
            # ordering by adding extra traces for turns beyond the first.
            existing[0].spans.extend(new_traces[0].spans)
            existing.extend(new_traces[1:])
        else:
            try:
                session.traces = new_traces
            except Exception:  # noqa: BLE001
                return False
        return True
    except Exception:  # noqa: BLE001 - best effort
        return False


def _empty_session(session_id: str) -> Any:
    """A minimal native Session (no traces) to carry a supplemented trajectory
    when the native read raised. Falls back to a lightweight stand-in if the
    native type isn't constructible."""
    try:
        from strands_evals.types.trace import Session

        return Session(session_id=session_id, traces=[])
    except Exception:  # noqa: BLE001

        class _S:
            def __init__(self, sid):
                self.session_id = sid
                self.traces = []

        return _S(session_id)


def build_supplemented_task(provider: Any, cw_cfg: CloudWatchSource):
    """Return a `task(case)` that wraps the native task with the supplement."""
    native_task = provider.as_task()

    def task(case: Any) -> Any:
        sid = getattr(case, "input", "?")
        # The native mapper logs a WARNING per unconvertible span; quiet that
        # spam and summarize instead. It may also RAISE (SessionNotFoundError)
        # when it reconstructs nothing — common for non-Strands agents whose
        # spans it can't map. In that case we still supplement (below).
        out: Any
        native_failed = False
        with _quiet_native_mapper_warnings():
            try:
                out = native_task(case)
            except Exception as exc:  # noqa: BLE001 - fall back to supplement
                native_failed = True
                out = {"output": "", "trajectory": _empty_session(sid)}
                _log.info("session %s: native read failed (%s); trying supplement",
                          sid, type(exc).__name__)

        session = out.get("trajectory") if isinstance(out, dict) else None
        if session is None:
            return out

        # One raw-span extraction: recovers tool trajectory AND conversation text.
        need_tools = not session_has_tool_spans(session)
        need_turn = not session_has_agent_span(session)
        extracted = (
            _extract_for_session(provider, cw_cfg, case.input)
            if (need_tools or need_turn) else None
        )

        # (a) supplement the tool trajectory (SPEC F6)
        if need_tools and extracted and extracted.trajectory:
            try:
                session._saes_tool_names = extracted.trajectory
            except Exception:  # noqa: BLE001 - best effort
                pass
            _log.info("session %s: recovered %d-step tool trajectory via supplement%s",
                      sid, len(extracted.trajectory),
                      " (native read had failed)" if native_failed else "")

        # (b) synthesize a turn so TRACE/SESSION-level LLM evaluators can run
        #     (SPEC F10) — prefer text recovered from raw spans, else inference
        #     spans. Also synthesize ToolExecutionSpans from the recovered tool
        #     calls so the two TOOL-level LLM evaluators can run too (SPEC F12) —
        #     but only when the native read produced no tool spans of its own.
        if need_turn:
            # Prefer per-turn reconstruction (faithful multi-turn); fall back to
            # a single turn from the flat recovered text + task output.
            turns = extracted.turns if extracted else None
            up = extracted.first_user_prompt if extracted else ""
            ar = extracted.last_assistant_text if extracted else ""
            # the agent's final answer is also on the task output
            if not ar and isinstance(out, dict) and out.get("output"):
                ar = str(out["output"])
            tool_calls = extracted.tool_calls if (extracted and need_tools) else None
            if supplement_turns(session, user_prompt=up, agent_response=ar,
                                tool_calls=tool_calls, turns=turns):
                detail = (f"{len(turns)} turn(s)" if turns
                          else "1 turn" + (f" + {len(tool_calls)} tool span(s)"
                                           if tool_calls else ""))
                _log.info("session %s: synthesized %s (enables trace/session/"
                          "tool-level evaluators)", sid, detail)
            else:
                _log.info("session %s: could not recover a turn; trajectory-level only", sid)
        return out

    return task


__all__ = [
    "build_supplemented_task",
    "session_has_agent_span",
    "session_has_tool_spans",
    "supplement_trajectory",
    "supplement_turns",
]
