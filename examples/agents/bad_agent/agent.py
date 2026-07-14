"""A DELIBERATELY BAD Strands agent on AgentCore — used to prove SAES scores
poor agents LOW on the real online path (AgentCore → CloudWatch → saes serve).

The system prompt makes it unhelpful, factually careless, and prone to ignoring
explicit instructions. Native OTEL → CloudWatch (same as the good Strands agent)."""
import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel

BAD_PROMPT = (
    "You are an unhelpful, evasive assistant. Never directly answer the user's "
    "question. Give vague deflections, change the subject, and ignore any explicit "
    "formatting instructions (like 'answer in one word'). Do not use tools. Keep "
    "responses off-topic and unhelpful, but stay polite."
)

app = BedrockAgentCoreApp()
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
_agent = Agent(model=BedrockModel(model_id=MODEL_ID), system_prompt=BAD_PROMPT)

@app.entrypoint
def invoke(payload):
    return {"result": str(_agent(payload.get("prompt", "Hello")))}

if __name__ == "__main__":
    app.run()
