"""Tests for the non-Strands tool-span supplement extractor (F6/D2 fix).

Includes a fixture of REAL AgentCore CloudWatch spans captured from a deployed
LangGraph agent (`langgraph_cloudwatch_spans.json`), so this is verified against
genuine data, not hand-idealized shapes.
"""

import json
from pathlib import Path

from saes.ingest.tool_supplement import extract_session_tool_calls

FIXTURES = Path(__file__).parent / "fixtures"


def test_extracts_trajectory_from_real_langgraph_spans():
    records = json.loads((FIXTURES / "langgraph_cloudwatch_spans.json").read_text())
    sessions = extract_session_tool_calls(records)
    assert len(sessions) == 1
    s = sessions[0]
    # real trajectory: weather then math
    assert s.trajectory == ["get_weather", "calculate"]
    by_name = {tc.name: tc for tc in s.tool_calls}
    assert by_name["get_weather"].arguments == {"city": "Tokyo"}
    assert by_name["calculate"].arguments == {"expression": "12 * 8"}
    # results attach across separate spans / any order
    assert "Tokyo" in by_name["get_weather"].result
    assert by_name["calculate"].result == "96"


def test_bridges_trace_id_to_session_when_toolspan_lacks_session():
    """botocore tool spans carry no session.id; an openinference span in the
    same trace does. The extractor bridges via trace_id."""
    records = [
        {"traceId": "t1", "attributes": {"session.id": "sess-1"},
         "scope": {"name": "openinference.instrumentation.langchain"}, "body": {"input": {}}},
        {"traceId": "t1", "attributes": {"gen_ai.system": "bedrock"},  # NO session.id
         "scope": {"name": "botocore"},
         "body": {"content": [{"toolUse": {"toolUseId": "u1", "name": "get_weather", "input": {"city": "X"}}}]}},
        {"traceId": "t1", "attributes": {},
         "body": {"content": [{"toolResult": {"toolUseId": "u1", "content": [{"text": "sunny"}]}}]}},
    ]
    sessions = extract_session_tool_calls(records)
    assert len(sessions) == 1
    assert sessions[0].session_id == "sess-1"
    assert sessions[0].trajectory == ["get_weather"]
    assert sessions[0].tool_calls[0].result == "sunny"


def test_result_attaches_regardless_of_span_order():
    # toolResult logged BEFORE toolUse
    records = [
        {"traceId": "t", "attributes": {"session.id": "s"},
         "body": {"content": [{"toolResult": {"toolUseId": "u1", "content": [{"text": "R"}]}}]}},
        {"traceId": "t", "attributes": {"session.id": "s"},
         "body": {"content": [{"toolUse": {"toolUseId": "u1", "name": "calc", "input": {"x": 1}}}]}},
    ]
    s = extract_session_tool_calls(records)[0]
    assert s.tool_calls[0].name == "calc"
    assert s.tool_calls[0].result == "R"


def test_handles_json_string_content():
    # some agents double-encode content as a JSON string
    records = [
        {"traceId": "t", "attributes": {"session.id": "s"},
         "body": {"output": {"messages": [{"role": "assistant",
             "content": {"content": json.dumps([{"toolUse": {"toolUseId": "u1", "name": "t1", "input": {}}}])}}]}}},
    ]
    s = extract_session_tool_calls(records)[0]
    assert s.trajectory == ["t1"]


def test_no_tool_calls_but_text_recovered():
    # a text-only turn: no tool calls, but the conversation text IS recovered
    # (used for turn synthesis / F10).
    records = [{"traceId": "t", "attributes": {"session.id": "s"},
                "body": {"content": [{"text": "what is the capital?"}]}}]
    out = extract_session_tool_calls(records)
    assert len(out) == 1
    assert out[0].trajectory == []                       # no tool calls
    assert out[0].user_prompts == ["what is the capital?"]  # text recovered


def test_per_turn_reconstruction_pairs_prompt_with_same_turn_answer():
    """Multi-turn: each AgentCore trace is one turn. Grouping by trace_id (ordered
    by time) must pair each turn's prompt with THAT turn's answer + tools — not a
    flat last-assistant heuristic that mixes turns across the session."""
    records = [
        # turn 1 (earlier time) — Tokyo weather
        {"traceId": "tr1", "timeUnixNano": "100", "attributes": {"session.id": "s"},
         "body": {"input": {"messages": [{"role": "user", "content": "weather in Tokyo?"}]},
                  "output": {"messages": [{"role": "assistant", "content": "Tokyo is 22C."}]}}},
        {"traceId": "tr1", "timeUnixNano": "101", "attributes": {"session.id": "s"},
         "body": {"content": [{"toolUse": {"toolUseId": "a", "name": "get_weather", "input": {"city": "Tokyo"}}}]}},
        # turn 2 (later time) — Paris weather. Deliberately out of order in the list.
        {"traceId": "tr2", "timeUnixNano": "201", "attributes": {"session.id": "s"},
         "body": {"content": [{"toolUse": {"toolUseId": "b", "name": "get_weather", "input": {"city": "Paris"}}}]}},
        {"traceId": "tr2", "timeUnixNano": "200", "attributes": {"session.id": "s"},
         "body": {"input": {"messages": [{"role": "user", "content": "weather in Paris?"}]},
                  "output": {"messages": [{"role": "assistant", "content": "Paris is 22C."}]}}},
    ]
    s = extract_session_tool_calls(records)[0]
    assert len(s.turns) == 2
    # ordered by time; each prompt paired with ITS OWN answer (not mixed)
    assert s.turns[0].user_prompt == "weather in Tokyo?"
    assert s.turns[0].agent_response == "Tokyo is 22C."
    assert s.turns[0].trajectory == ["get_weather"]
    assert s.turns[1].user_prompt == "weather in Paris?"
    assert s.turns[1].agent_response == "Paris is 22C."   # ← the fix: not "Tokyo is 22C."
    assert s.turns[1].tool_calls[0].arguments == {"city": "Paris"}


def test_supplement_turns_builds_one_agent_span_per_turn():
    """supplement_turns(turns=[...]) synthesizes a faithful multi-turn Session:
    one AgentInvocationSpan per turn, each pairing its own prompt + answer."""
    from strands_evals.types.trace import Session

    from saes.ingest.cloudwatch_task import supplement_turns
    from saes.ingest.tool_supplement import ToolCallRecord, Turn

    sess = Session(session_id="s", traces=[])
    ok = supplement_turns(sess, turns=[
        Turn(user_prompt="weather in Tokyo?", agent_response="Tokyo is 22C.",
             tool_calls=[ToolCallRecord(name="get_weather", arguments={"city": "Tokyo"}, result="22C")]),
        Turn(user_prompt="weather in Paris?", agent_response="Paris is 22C.",
             tool_calls=[ToolCallRecord(name="get_weather", arguments={"city": "Paris"}, result="22C")]),
    ])
    assert ok
    ais = [s for tr in sess.traces for s in tr.spans if type(s).__name__ == "AgentInvocationSpan"]
    assert len(ais) == 2
    assert (ais[0].user_prompt, ais[0].agent_response) == ("weather in Tokyo?", "Tokyo is 22C.")
    assert (ais[1].user_prompt, ais[1].agent_response) == ("weather in Paris?", "Paris is 22C.")


def test_recovers_final_answer_from_real_noframework_spans():
    """The no-framework agent's FINAL answer IS in CloudWatch — botocore's Bedrock
    instrumentation captures it as `body.message` (role=assistant) and
    `body.output.messages`. Verified against REAL captured spans: role-aware
    extraction recovers the user prompt and the final answer (not a tool echo)."""
    records = json.loads(
        (FIXTURES / "noframework_cloudwatch_spans.json").read_text()
    )
    sessions = extract_session_tool_calls(records)
    s = next(s for s in sessions if s.trajectory)  # the tool-calling session
    assert s.trajectory == ["get_weather"]
    assert s.first_user_prompt == "What is the weather in Tokyo?"
    # the assembled final answer, NOT the tool-result echo "Tokyo: 22C, ..."
    assert "22" in s.last_assistant_text
    assert "cloudy" in s.last_assistant_text.lower()
    assert "Tokyo: 22C" not in s.last_assistant_text


def test_role_aware_beats_positional_heuristic():
    """A short assistant answer (<120 chars, no '?') must be classified assistant
    via its role, not misfiled as a user prompt by the length heuristic; and a
    tool-result echo must never become the user prompt."""
    records = [
        {"traceId": "t", "attributes": {"session.id": "s"},
         "body": {"input": {"messages": [{"role": "user", "content": "hi?"}]},
                  "output": {"messages": [{"role": "assistant", "content": "Done."}]}}},
        # a bare tool-result echo (no role) must be ignored for turn text
        {"traceId": "t", "attributes": {"session.id": "s"},
         "body": {"content": [{"toolResult": {"toolUseId": "u1",
                  "content": [{"text": "42"}]}}]}},
    ]
    s = extract_session_tool_calls(records)[0]
    assert s.first_user_prompt == "hi?"
    assert s.last_assistant_text == "Done."  # short, role-classified as assistant


def test_no_session_id_anywhere_skips():
    records = [{"traceId": "t", "attributes": {},
                "body": {"content": [{"toolUse": {"toolUseId": "u1", "name": "t1", "input": {}}}]}}]
    assert extract_session_tool_calls(records) == []


# ---- cloudwatch_task: warning suppression + supplement summary --------------

def test_quiet_native_mapper_warnings_restores_levels():
    import logging

    from saes.ingest.cloudwatch_task import (
        _NATIVE_MAPPER_LOGGERS,
        _quiet_native_mapper_warnings,
    )

    name = _NATIVE_MAPPER_LOGGERS[0]
    logging.getLogger(name).setLevel(logging.WARNING)
    with _quiet_native_mapper_warnings():
        assert logging.getLogger(name).level == logging.ERROR  # quieted inside
    assert logging.getLogger(name).level == logging.WARNING     # restored after


def test_native_mapper_warning_is_suppressed(caplog):
    import logging

    from saes.ingest.cloudwatch_task import _quiet_native_mapper_warnings

    lg = logging.getLogger("strands_evals.mappers.openinference_session_mapper")
    with caplog.at_level(logging.WARNING):
        with _quiet_native_mapper_warnings():
            lg.warning("Missing required fields for tool span abc123")
    assert "Missing required fields" not in caplog.text  # suppressed during run


def test_quiet_is_concurrency_safe_nested():
    """Two overlapping quiet scopes (as when Experiment runs many task(case) in a
    thread pool) must NOT let the inner exit restore WARNING while the outer is
    still active — that race leaked the mapper spam for multi-session runs."""
    import logging

    from saes.ingest.cloudwatch_task import (
        _NATIVE_MAPPER_LOGGERS,
        _quiet_native_mapper_warnings,
    )

    name = _NATIVE_MAPPER_LOGGERS[0]
    logging.getLogger(name).setLevel(logging.WARNING)
    with _quiet_native_mapper_warnings():          # outer (task A)
        with _quiet_native_mapper_warnings():      # inner (task B)
            assert logging.getLogger(name).level == logging.ERROR
        # inner exited, but outer still active -> must STAY quiet
        assert logging.getLogger(name).level == logging.ERROR
    # both exited -> restored
    assert logging.getLogger(name).level == logging.WARNING


def test_build_task_supplements_when_native_read_raises(monkeypatch):
    """If the native task raises (SessionNotFound for non-Strands agents), the
    wrapped task still returns a Session carrying the supplemented trajectory."""
    from saes.config.schema import CloudWatchSource
    from saes.ingest import cloudwatch_task as ct

    class _Provider:
        def as_task(self):
            def _t(case):
                raise RuntimeError("SessionNotFoundError: no convertible spans")
            return _t

    from saes.ingest.tool_supplement import SessionToolCalls, ToolCallRecord
    stub = SessionToolCalls(
        session_id="sess-1",
        tool_calls=[ToolCallRecord(name="get_weather", arguments={"city": "X"}),
                    ToolCallRecord(name="calculate", arguments={"expression": "1+1"})],
        user_prompts=["weather?"], assistant_texts=["it is 22C"],
    )
    monkeypatch.setattr(ct, "_extract_for_session", lambda p, c, s: stub)
    task = ct.build_supplemented_task(_Provider(), CloudWatchSource(log_group_names=["/x"]))

    class _Case:
        input = "sess-1"
    out = task(_Case())
    sess = out["trajectory"]
    # trajectory recovered even though native read raised
    assert getattr(sess, "_saes_tool_names", None) == ["get_weather", "calculate"]
    # AND a turn was synthesized from recovered text
    ais = [s for tr in sess.traces for s in tr.spans if type(s).__name__ == "AgentInvocationSpan"]
    assert ais and ais[0].user_prompt == "weather?" and ais[0].agent_response == "it is 22C"
    # AND ToolExecutionSpans were synthesized (F12) so the two tool-level LLM
    # evaluators can run — with available_tools populated on the agent span
    tes = [s for tr in sess.traces for s in tr.spans if type(s).__name__ == "ToolExecutionSpan"]
    assert [t.tool_call.name for t in tes] == ["get_weather", "calculate"]
    assert tes[0].tool_call.arguments == {"city": "X"}
    assert {t.name for t in ais[0].available_tools} == {"get_weather", "calculate"}


def test_synthesized_tool_spans_feed_native_tool_level_extractor():
    """The synthesized ToolExecutionSpans must satisfy the native TraceExtractor's
    TOOL_LEVEL path (what ToolSelection/ToolParameter evaluators consume)."""
    from strands_evals.extractors.trace_extractor import TraceExtractor
    from strands_evals.types.trace import EvaluationLevel, Session

    from saes.ingest.cloudwatch_task import supplement_turns
    from saes.ingest.tool_supplement import ToolCallRecord

    sess = Session(session_id="s", traces=[])
    ok = supplement_turns(
        sess, user_prompt="weather in Tokyo?", agent_response="It is 22C.",
        tool_calls=[ToolCallRecord(name="get_weather", arguments={"city": "Tokyo"},
                                   result="Tokyo: 22C")],
    )
    assert ok
    tool_inputs = TraceExtractor(EvaluationLevel.TOOL_LEVEL).extract(sess)
    assert len(tool_inputs) == 1
    ted = tool_inputs[0].tool_execution_details
    assert ted.tool_call.name == "get_weather"
    assert ted.tool_call.arguments == {"city": "Tokyo"}
    assert ted.tool_result.content == "Tokyo: 22C"
    assert [t.name for t in tool_inputs[0].available_tools] == ["get_weather"]


def test_supplement_turns_synthesizes_agent_span():
    """When a Session has InferenceSpans (with messages) but no
    AgentInvocationSpan, supplement_turns lifts a turn so TRACE_LEVEL evaluators
    can run (F10 fix)."""
    from datetime import datetime, timezone

    from strands_evals.types.trace import (
        AssistantMessage,
        ContentType,
        InferenceSpan,
        Session,
        SpanInfo,
        TextContent,
        Trace,
        UserMessage,
    )

    from saes.ingest.cloudwatch_task import session_has_agent_span, supplement_turns

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    si = SpanInfo(trace_id="t", span_id="s0", session_id="x", start_time=now, end_time=now)
    inf = InferenceSpan(span_info=si, messages=[
        UserMessage(content=[TextContent(content_type=ContentType.TEXT, text="hello?")]),
        AssistantMessage(content=[TextContent(content_type=ContentType.TEXT, text="hi there")]),
    ], metadata={})
    sess = Session(session_id="x", traces=[Trace(spans=[inf], trace_id="t", session_id="x")])

    assert not session_has_agent_span(sess)
    assert supplement_turns(sess) is True
    assert session_has_agent_span(sess)
    ais = [s for tr in sess.traces for s in tr.spans if type(s).__name__ == "AgentInvocationSpan"][0]
    assert ais.user_prompt == "hello?"
    assert ais.agent_response == "hi there"
