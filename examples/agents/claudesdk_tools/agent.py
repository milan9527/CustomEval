"""Claude Agent SDK tool-calling agent on AgentCore, instrumented per the SAES
OTEL contract (custom-framework path).

Unlike the boto3-based agents, the Claude Agent SDK drives Bedrock from a
bundled CLI subprocess — AgentCore's botocore instrumentation never sees the
model calls, so nothing is captured for free. The contract is satisfied by
hand:

  - one root span per invocation (= one trace per turn) carrying
    `session.id`, `gen_ai.prompt`, and `gen_ai.completion`;
  - per tool call, a child span (`gen_ai.tool.name`) plus OTEL events whose
    body carries Bedrock-Converse-shaped `toolUse`/`toolResult` blocks — the
    exact shape SAES's tool supplement recovers for non-Strands agents.
"""
import ast
import operator as op
import os
import uuid

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    create_sdk_mcp_server,
    query,
    tool,
)
from opentelemetry import trace
from opentelemetry._events import Event, get_event_logger

tracer = trace.get_tracer("claudesdk.agent")
events = get_event_logger("claudesdk.agent")

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
SYSTEM = "You are a helpful assistant. Use tools for weather and math."

# SpanContext of the current invocation's root span. Tool callbacks run inside
# the SDK's own asyncio tasks, so parent them explicitly rather than relying on
# contextvar propagation through the CLI transport.
_ROOT_CTX = None


def get_weather(city):
    return f"{city}: 22C, partly cloudy, light wind."


def calculate(expression):
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
           ast.Pow: op.pow, ast.USub: op.neg}
    def ev(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp): return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp): return ops[type(n.op)](ev(n.operand))
        raise ValueError()
    return str(ev(ast.parse(expression, mode="eval").body))


def _traced_tool(name, args, fn):
    """Run a tool under a child span of the invocation root and emit the
    Converse-shaped toolUse/toolResult event pair the SAES supplement reads."""
    parent = (trace.set_span_in_context(trace.NonRecordingSpan(_ROOT_CTX))
              if _ROOT_CTX else None)
    with tracer.start_as_current_span(f"tool.{name}", context=parent) as span:
        span.set_attribute("gen_ai.tool.name", name)
        result = fn(**args)
        ctx = span.get_span_context()
        tool_use_id = f"tooluse_{uuid.uuid4().hex[:16]}"
        events.emit(Event(
            name="gen_ai.tool.request", trace_id=ctx.trace_id, span_id=ctx.span_id,
            body={"content": [{"toolUse": {
                "toolUseId": tool_use_id, "name": name, "input": args}}]},
        ))
        events.emit(Event(
            name="gen_ai.tool.result", trace_id=ctx.trace_id, span_id=ctx.span_id,
            body={"content": [{"toolResult": {
                "toolUseId": tool_use_id, "content": [{"text": result}]}}]},
        ))
    return result


@tool("get_weather", "Weather for a city", {"city": str})
async def weather_tool(args):
    res = _traced_tool("get_weather", {"city": args["city"]}, get_weather)
    return {"content": [{"type": "text", "text": res}]}


@tool("calculate", "Evaluate an arithmetic expression", {"expression": str})
async def calculate_tool(args):
    res = _traced_tool("calculate", {"expression": args["expression"]}, calculate)
    return {"content": [{"type": "text", "text": res}]}


toolbox = create_sdk_mcp_server(name="toolbox", tools=[weather_tool, calculate_tool])

OPTIONS = ClaudeAgentOptions(
    system_prompt=SYSTEM,
    model=MODEL_ID,
    mcp_servers={"toolbox": toolbox},
    allowed_tools=["mcp__toolbox__get_weather", "mcp__toolbox__calculate"],
    disallowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep",
                      "WebFetch", "WebSearch", "Task", "TodoWrite", "NotebookEdit"],
    permission_mode="bypassPermissions",
    max_turns=8,
    env={
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
    },
)


async def run_agent(prompt):
    answer = ""
    async for message in query(prompt=prompt, options=OPTIONS):
        if isinstance(message, ResultMessage):
            answer = message.result or answer
    return answer


app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload, context):
    global _ROOT_CTX
    prompt = payload.get("prompt", "Hello")
    session_id = getattr(context, "session_id", None) or str(uuid.uuid4())
    # one trace per turn, carrying the three contract attributes
    with tracer.start_as_current_span("agent.invocation") as span:
        span.set_attribute("session.id", session_id)          # contract item 1
        span.set_attribute("gen_ai.prompt", prompt)           # contract item 2
        events.emit(Event(
            name="gen_ai.user.message",
            body={"message": {"role": "user", "content": [{"text": prompt}]}},
        ))
        _ROOT_CTX = span.get_span_context()
        try:
            answer = await run_agent(prompt)
        finally:
            _ROOT_CTX = None
        span.set_attribute("gen_ai.completion", answer)       # contract item 3
        events.emit(Event(
            name="gen_ai.assistant.message",
            body={"message": {"role": "assistant", "content": [{"text": answer}]}},
        ))
    return {"result": answer}


if __name__ == "__main__":
    app.run()
