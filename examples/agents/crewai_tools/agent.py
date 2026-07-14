"""CrewAI tool-calling agent on AgentCore, instrumented with OpenInference
(CrewAI instrumentor) so OTEL spans export to CloudWatch."""
import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from openinference.instrumentation.crewai import CrewAIInstrumentor
CrewAIInstrumentor().instrument()
# CrewAI uses LiteLLM under the hood; instrument that too for LLM spans
try:
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    LiteLLMInstrumentor().instrument()
except Exception:
    pass

from crewai import Agent as CrewAgent, Task, Crew
from crewai.tools import tool

@tool("get_weather")
def get_weather(city: str) -> str:
    """Get the current weather forecast for a city."""
    return f"{city}: 22C, partly cloudy, light wind."

@tool("calculate")
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
assistant = CrewAgent(role="Assistant", goal="Answer using tools",
    backstory="You use weather and math tools to help users.",
    llm=f"bedrock/{MODEL_ID}", tools=[get_weather, calculate], verbose=False)

app = BedrockAgentCoreApp()
@app.entrypoint
def invoke(payload):
    prompt = payload.get("prompt", "Hello")
    task = Task(description=prompt, expected_output="A helpful answer.", agent=assistant)
    crew = Crew(agents=[assistant], tasks=[task], verbose=False)
    return {"result": str(crew.kickoff())}

if __name__ == "__main__":
    app.run()
