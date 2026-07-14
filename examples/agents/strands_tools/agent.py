"""Strands tool-calling agent on AgentCore. Native OTEL -> CloudWatch."""
import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel

@tool
def get_weather(city: str) -> str:
    """Get the current weather forecast for a city."""
    return f"{city}: 22C, partly cloudy, light wind."

@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '15/100*240'."""
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}
    def ev(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp): return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp): return ops[type(n.op)](ev(n.operand))
        raise ValueError("bad expr")
    return str(ev(ast.parse(expression, mode="eval").body))

app = BedrockAgentCoreApp()
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
_agent = Agent(model=BedrockModel(model_id=MODEL_ID), tools=[get_weather, calculate],
               system_prompt="You are a helpful assistant. Use tools for weather and math.")

@app.entrypoint
def invoke(payload):
    return {"result": str(_agent(payload.get("prompt", "Hello")))}

if __name__ == "__main__":
    app.run()
