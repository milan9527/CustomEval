"""Supplementary tool-span extractor for non-Strands CloudWatch traces (F6/D2).

Background (verified against real AgentCore CloudWatch logs from LangGraph and
other non-Strands agents): the native `strands-agents-evals` CloudWatch mapper
extracts tool calls only from the Strands-shaped `body.output.messages[].content`
Converse blocks. Non-Strands agents on AgentCore instead emit:

  - `botocore.bedrock-runtime` spans whose `body.content` carries real Bedrock
    Converse `toolUse` (request) and `toolResult` blocks — BUT no `session.id`;
  - `openinference.*` spans in the SAME `trace_id` that DO carry `session.id`.

So tool calls are present in CloudWatch but the native mapper misses them,
leaving non-Strands agents evaluable only at trace level. This module bridges
the gap purely on the SAES side (no change to the agent, no reverse-engineering
of AgentCore's telemetry): it reads the raw CloudWatch span records, links
tool-bearing spans to a session via shared `trace_id`, and returns the tool
trajectory. SAES uses it to enrich a session's trajectory when the native
mapper produced none.

This is an ingestion adapter (SAES owns ingestion) — it does NOT reimplement
evaluators or the native Session model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    tool_call_id: str | None = None
    result: str | None = None


@dataclass
class Turn:
    """One reconstructed conversation turn (one AgentCore trace). Pairs the
    user's prompt with the agent's final answer and the tools used *in that
    turn*, so multi-turn sessions score turn-by-turn instead of mixing turns."""

    user_prompt: str = ""
    agent_response: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    @property
    def trajectory(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]


@dataclass
class SessionToolCalls:
    session_id: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    # conversation text recovered from raw spans (for turn synthesis / F10):
    user_prompts: list[str] = field(default_factory=list)
    assistant_texts: list[str] = field(default_factory=list)
    # per-turn reconstruction (one entry per AgentCore trace, time-ordered).
    # Empty when no roled text/trace grouping was available (falls back to the
    # flat fields above). Used to synthesize a faithful multi-turn Session.
    turns: list[Turn] = field(default_factory=list)

    @property
    def trajectory(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]

    @property
    def first_user_prompt(self) -> str:
        return self.user_prompts[0] if self.user_prompts else ""

    @property
    def last_assistant_text(self) -> str:
        return self.assistant_texts[-1] if self.assistant_texts else ""


def _flat_attrs(rec: dict[str, Any]) -> dict[str, Any]:
    a = rec.get("attributes")
    if isinstance(a, dict):
        return a
    if isinstance(a, list):
        out: dict[str, Any] = {}
        for item in a:
            if isinstance(item, dict) and "key" in item:
                v = item.get("value")
                if isinstance(v, dict):
                    v = next(iter(v.values()), None)
                out[item["key"]] = v
        return out
    return {}


def _session_id_of(rec: dict[str, Any]) -> str | None:
    a = _flat_attrs(rec)
    for k in ("session.id", "gen_ai.session.id", "session_id"):
        if a.get(k):
            return str(a[k])
    return None


def _content_blocks(body: Any) -> list[dict[str, Any]]:
    """Converse content blocks from a span body, tolerant of shapes:
    - {"content": [ ...blocks... ]}
    - {"content": "<json-str of blocks>"}
    - {"output": {"messages": [{"content": {"content": "<json-str>"}}]}}
    """
    if not isinstance(body, dict):
        return []
    blocks: list[dict[str, Any]] = []

    def _coerce(c: Any) -> list[dict[str, Any]]:
        if isinstance(c, list):
            return [b for b in c if isinstance(b, dict)]
        if isinstance(c, str):
            try:
                parsed = json.loads(c)
            except (json.JSONDecodeError, TypeError):
                return []
            return [b for b in parsed if isinstance(b, dict)] if isinstance(parsed, list) else []
        return []

    # direct content
    if "content" in body:
        blocks.extend(_coerce(body["content"]))
    # nested output.messages[].content(.content)
    for section in ("output", "input"):
        sec = body.get(section)
        msgs = sec.get("messages", []) if isinstance(sec, dict) else []
        for m in msgs:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, dict):
                c = c.get("content") or c.get("message")
            blocks.extend(_coerce(c))
    return blocks


_SYSTEMISH = ("you are ", "system:")


def _is_systemish(text: str) -> bool:
    low = text.strip().lower()
    return any(low.startswith(s) for s in _SYSTEMISH)


def _texts_from_content(content: Any) -> list[str]:
    """Plain-text strings from a message's `content`, tolerant of shapes:
    a bare string, a JSON-string of blocks, a list of Converse blocks, or a
    nested {"content": ...} wrapper. Ignores non-text blocks (toolUse/toolResult)."""
    if content is None:
        return []
    if isinstance(content, str):
        s = content.strip()
        if s[:1] in ("[", "{"):
            try:
                parsed = json.loads(s)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                return [b["text"] for b in parsed
                        if isinstance(b, dict) and isinstance(b.get("text"), str)]
            if isinstance(parsed, dict):
                return _texts_from_content(parsed)
        return [s] if s else []
    if isinstance(content, list):
        return [b["text"] for b in content
                if isinstance(b, dict) and isinstance(b.get("text"), str)]
    if isinstance(content, dict):
        if "content" in content:
            return _texts_from_content(content["content"])
        if isinstance(content.get("text"), str):
            return [content["text"]]
    return []


def _iter_role_texts(body: Any):
    """Yield (role, text) for roled messages in a span body. Handles:
    - `body.message` (botocore Bedrock response: role=assistant, finish_reason)
    - `body.input.messages[]`  (role defaults to 'user')
    - `body.output.messages[]` (role defaults to 'assistant')
    Verified against real no-framework AgentCore CloudWatch spans — this is where
    the agent's FINAL answer actually lives (botocore captures it)."""
    if not isinstance(body, dict):
        return
    msg = body.get("message")
    if isinstance(msg, dict):
        role = msg.get("role") or "assistant"
        for t in _texts_from_content(msg.get("content")):
            yield role, t
    for section, default_role in (("input", "user"), ("output", "assistant")):
        sec = body.get(section)
        if not isinstance(sec, dict):
            continue
        for m in sec.get("messages", []) or []:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or default_role
            for t in _texts_from_content(m.get("content")):
                yield role, t


def _trace_id_of(rec: dict[str, Any]) -> str | None:
    return rec.get("traceId") or rec.get("trace_id")


def _rec_time(rec: dict[str, Any]) -> int:
    """Best-effort record timestamp (ns) for ordering turns within a session."""
    for k in ("timeUnixNano", "observedTimeUnixNano", "startTimeUnixNano"):
        v = rec.get(k)
        if v:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return 0


def extract_session_tool_calls(records: list[dict[str, Any]]) -> list[SessionToolCalls]:
    """Bridge trace_id -> session.id and extract Converse tool calls from all
    tool-bearing spans. Returns one SessionToolCalls per discovered session.

    Also reconstructs a **per-turn** view (`SessionToolCalls.turns`): each
    AgentCore trace is one conversation turn, so grouping by trace_id (ordered by
    time) pairs each user prompt with the answer + tools of that same turn —
    essential for faithful multi-turn scoring (a flat last-assistant heuristic
    mixes turns)."""
    # 1. map trace_id -> session_id from any span that carries a session id
    trace_to_session: dict[str, str] = {}
    for rec in records:
        tid = _trace_id_of(rec)
        sid = _session_id_of(rec)
        if tid and sid and tid not in trace_to_session:
            trace_to_session[tid] = sid

    def _sid_for(rec: dict[str, Any]) -> str | None:
        tid = rec.get("traceId") or rec.get("trace_id")
        return trace_to_session.get(tid) if tid else _session_id_of(rec)

    # 2. PASS 1 — collect toolUse (requests), preserving order, per session
    calls: dict[str, dict[str, ToolCallRecord]] = {}
    order: dict[str, list[str]] = {}
    for rec in records:
        sid = _sid_for(rec)
        if not sid:
            continue
        for block in _content_blocks(rec.get("body")):
            if "toolUse" not in block:
                continue
            tu = block["toolUse"]
            cid = tu.get("toolUseId") or f"_{len(order.get(sid, []))}"
            calls.setdefault(sid, {})
            order.setdefault(sid, [])
            if cid not in calls[sid]:
                calls[sid][cid] = ToolCallRecord(
                    name=tu.get("name", ""), arguments=tu.get("input", {}) or {}, tool_call_id=cid
                )
                order[sid].append(cid)

    # 3. PASS 2 — attach toolResult by id (results live on separate spans that
    #    may be logged before OR after the toolUse span)
    for rec in records:
        sid = _sid_for(rec)
        if not sid or sid not in calls:
            continue
        for block in _content_blocks(rec.get("body")):
            if "toolResult" not in block:
                continue
            trb = block["toolResult"]
            cid = trb.get("toolUseId")
            text = next(
                (c["text"] for c in trb.get("content", []) if isinstance(c, dict) and "text" in c),
                "",
            )
            if cid in calls[sid]:
                calls[sid][cid].result = text

    # 4. PASS 3 — collect conversation TEXT per session for turn synthesis (F10).
    #    Verified against REAL no-framework AgentCore spans: botocore's Bedrock
    #    instrumentation DOES capture the final answer — as `body.message`
    #    (role=assistant, finish_reason=end_turn) and/or `body.{input,output}.
    #    messages[]` with explicit roles. So we extract text ROLE-AWARE from those
    #    shapes (authoritative), plus the agent's own gen_ai.prompt/completion
    #    attributes. Only if a session yields NO roled text do we fall back to the
    #    old positional heuristic on bare `content` blocks.
    user_texts: dict[str, list[str]] = {}
    asst_texts: dict[str, list[str]] = {}
    heur_texts: dict[str, list[str]] = {}
    for rec in records:
        sid = _sid_for(rec)
        if not sid:
            continue
        attrs = _flat_attrs(rec)
        # explicit agent-level prompt/completion attributes (e.g. no-framework agent)
        if attrs.get("gen_ai.prompt"):
            user_texts.setdefault(sid, []).append(str(attrs["gen_ai.prompt"]))
        if attrs.get("gen_ai.completion"):
            asst_texts.setdefault(sid, []).append(str(attrs["gen_ai.completion"]))
        # authoritative: roled messages (body.message / body.{input,output}.messages)
        for role, txt in _iter_role_texts(rec.get("body")):
            if _is_systemish(txt):
                continue
            (user_texts if role == "user" else asst_texts).setdefault(sid, []).append(txt)
        # fallback pool: bare content text blocks (no role) — heuristic, used only
        # when this session produced no roled text at all
        for block in _content_blocks(rec.get("body")):
            txt = block.get("text") if isinstance(block, dict) else None
            if isinstance(txt, str) and txt.strip() and not _is_systemish(txt):
                heur_texts.setdefault(sid, []).append(txt)

    # 5. PASS 4 — per-turn reconstruction. Each AgentCore trace is one turn;
    #    group text + tool calls by trace_id, order traces by time, and pair each
    #    turn's user prompt with that turn's own answer + tools.
    turns_by_sid = _reconstruct_turns(records, _sid_for)

    all_sids = set(order) | set(user_texts) | set(asst_texts) | set(heur_texts)
    result = []
    for sid in all_sids:
        users = user_texts.get(sid, [])
        assts = asst_texts.get(sid, [])
        # Only fall back to the positional heuristic when this session produced no
        # roled text at all (older/odd agents that emit bare `content` blocks with
        # no role and no roled message wrapper).
        if not users and not assts:
            for txt in heur_texts.get(sid, []):
                (users if txt.strip().endswith("?") or len(txt) < 120 else assts).append(txt)
        result.append(SessionToolCalls(
            session_id=sid,
            tool_calls=[calls[sid][i] for i in order.get(sid, [])],
            user_prompts=users,
            assistant_texts=assts,
            turns=turns_by_sid.get(sid, []),
        ))
    return result


def _reconstruct_turns(records, sid_for) -> dict[str, list[Turn]]:
    """Group records by (session, trace), one Turn per trace, ordered by time.

    Within a trace: first non-system user text -> prompt; last assistant text ->
    answer; toolUse blocks (with results attached) -> that turn's tool calls."""
    # session -> trace -> {"time", "users", "assts", "tools"(id->rec), "order"}
    sessions: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in records:
        sid = sid_for(rec)
        tid = _trace_id_of(rec)
        if not sid or not tid:
            continue
        traces = sessions.setdefault(sid, {})
        tr = traces.setdefault(tid, {"time": _rec_time(rec), "users": [], "assts": [],
                                     "tools": {}, "order": []})
        tr["time"] = min(tr["time"] or _rec_time(rec), _rec_time(rec) or tr["time"])
        for role, txt in _iter_role_texts(rec.get("body")):
            if _is_systemish(txt):
                continue
            (tr["users"] if role == "user" else tr["assts"]).append(txt)
        for block in _content_blocks(rec.get("body")):
            if "toolUse" in block:
                tu = block["toolUse"]
                cid = tu.get("toolUseId") or f"_{len(tr['order'])}"
                if cid not in tr["tools"]:
                    tr["tools"][cid] = ToolCallRecord(
                        name=tu.get("name", ""), arguments=tu.get("input", {}) or {},
                        tool_call_id=cid)
                    tr["order"].append(cid)
            if "toolResult" in block:
                trb = block["toolResult"]
                cid = trb.get("toolUseId")
                text = next((c["text"] for c in trb.get("content", [])
                             if isinstance(c, dict) and "text" in c), "")
                if cid in tr["tools"]:
                    tr["tools"][cid].result = text

    out: dict[str, list[Turn]] = {}
    for sid, traces in sessions.items():
        turns: list[Turn] = []
        for tid in sorted(traces, key=lambda t: traces[t]["time"]):
            tr = traces[tid]
            up = tr["users"][0] if tr["users"] else ""
            ar = tr["assts"][-1] if tr["assts"] else ""
            tools = [tr["tools"][c] for c in tr["order"]]
            if up or ar or tools:
                turns.append(Turn(user_prompt=up, agent_response=ar, tool_calls=tools))
        if turns:
            out[sid] = turns
    return out


__all__ = ["SessionToolCalls", "ToolCallRecord", "Turn", "extract_session_tool_calls"]
