"""No-framework tool-calling agent on AgentCore: plain Python + Bedrock Converse
(native tool use). NO telemetry code — this is deliberately a "bare" agent.

The point: AgentCore's botocore instrumentation already captures the full turn
(user prompt, tool calls/results, AND the final assembled answer) as OTEL spans
in CloudWatch. SAES's ingestion supplement reconstructs the evaluation turn from
those standard spans — so this agent needs zero SAES-specific instrumentation.
"""
import ast
import operator as op
import os

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from opentelemetry import trace

tracer = trace.get_tracer("noframework.agent")

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
br = boto3.client("bedrock-runtime", region_name="us-east-1")


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


TOOLS = [
    {"toolSpec": {"name": "get_weather", "description": "Weather for a city",
                  "inputSchema": {"json": {"type": "object",
                                           "properties": {"city": {"type": "string"}},
                                           "required": ["city"]}}}},
    {"toolSpec": {"name": "calculate", "description": "Evaluate arithmetic",
                  "inputSchema": {"json": {"type": "object",
                                           "properties": {"expression": {"type": "string"}},
                                           "required": ["expression"]}}}},
]
FN = {"get_weather": lambda a: get_weather(**a), "calculate": lambda a: calculate(**a)}
SYSTEM = "You are a helpful assistant. Use tools for weather and math."


def run_agent(prompt):
    msgs = [{"role": "user", "content": [{"text": prompt}]}]
    for _ in range(5):
        r = br.converse(modelId=MODEL_ID, system=[{"text": SYSTEM}], messages=msgs,
                        toolConfig={"tools": TOOLS})
        out = r["output"]["message"]; msgs.append(out)
        tuses = [c for c in out["content"] if "toolUse" in c]
        if r["stopReason"] != "tool_use":
            return "".join(c.get("text", "") for c in out["content"])
        results = []
        for c in tuses:
            tu = c["toolUse"]
            res = FN[tu["name"]](tu["input"])
            results.append({"toolResult": {"toolUseId": tu["toolUseId"], "content": [{"text": res}]}})
        msgs.append({"role": "user", "content": results})
    return "(max turns)"


app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload):
    prompt = payload.get("prompt", "Hello")
    with tracer.start_as_current_span("agent.invocation"):
        answer = run_agent(prompt)
    return {"result": answer}


if __name__ == "__main__":
    app.run()
