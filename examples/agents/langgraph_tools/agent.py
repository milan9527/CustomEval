"""LangGraph tool-calling agent on AgentCore, instrumented with OpenInference
so OTEL spans (incl. tool calls) export to CloudWatch."""
import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# --- OpenInference instrumentation for LangChain/LangGraph ---
from openinference.instrumentation.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument()

from langchain_core.tools import tool
from langchain_aws import ChatBedrockConverse
from langgraph.prebuilt import create_react_agent

@tool
def get_weather(city: str) -> str:
    """Get the current weather forecast for a city."""
    return f"{city}: 22C, partly cloudy, light wind."

@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression."""
    import ast, operator as op
    ops={ast.Add:op.add,ast.Sub:op.sub,ast.Mult:op.mul,ast.Div:op.truediv,ast.Pow:op.pow,ast.USub:op.neg}
    def ev(n):
        if isinstance(n,ast.Constant): return n.value
        if isinstance(n,ast.BinOp): return ops[type(n.op)](ev(n.left),ev(n.right))
        if isinstance(n,ast.UnaryOp): return ops[type(n.op)](ev(n.operand))
        raise ValueError()
    return str(ev(ast.parse(expression,mode="eval").body))

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
llm = ChatBedrockConverse(model=MODEL_ID, region_name="us-east-1", provider="anthropic")
graph = create_react_agent(llm, tools=[get_weather, calculate])

app = BedrockAgentCoreApp()
@app.entrypoint
def invoke(payload):
    prompt = payload.get("prompt", "Hello")
    result = graph.invoke({"messages": [{"role": "user", "content": prompt}]})
    return {"result": result["messages"][-1].content}

if __name__ == "__main__":
    app.run()
