"""RAG knowledge-assistant agent on AgentCore (Strands, native OTEL→CloudWatch).

Answers HR-policy questions grounded in a small retrievable knowledge base. The
system prompt forbids answering beyond retrieved context (tests Faithfulness /
ContextRelevance / Correctness). Retrieval is deterministic for reproducibility."""
import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel

# --- tiny knowledge base (the ground truth the agent must stay faithful to) --
_KB = {
    "pto": ("Full-time employees accrue 20 days of paid time off per year, "
            "accrued monthly. Unused PTO carries over up to 5 days into the next year."),
    "remote": ("Employees may work remotely up to 3 days per week with manager "
               "approval. Fully-remote roles are approved case-by-case by HR."),
    "expenses": ("Business expenses under $75 need no receipt; $75 and above "
                 "require an itemized receipt submitted within 30 days."),
    "parental": ("Parental leave is 16 weeks paid for the primary caregiver and "
                 "6 weeks paid for the secondary caregiver."),
}


@tool
def search_policy(query: str) -> str:
    """Search the HR policy knowledge base. Returns the most relevant passage(s)."""
    q = query.lower()
    hits = [v for k, v in _KB.items() if k in q or any(w in q for w in k.split())]
    if not hits:
        # keyword fallback
        for k, v in _KB.items():
            if any(term in q for term in (k, k[:4])):
                hits.append(v)
    return "\n---\n".join(hits) if hits else "NO_RESULTS"


SYSTEM = (
    "You are an HR policy assistant. You MUST answer only from passages returned by "
    "search_policy. Always call search_policy first. If it returns NO_RESULTS or the "
    "passage doesn't cover the question, say you don't have that policy and offer to "
    "escalate to HR — never guess or invent numbers. Quote the specific figures from "
    "the retrieved text."
)

app = BedrockAgentCoreApp()
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
_agent = Agent(model=BedrockModel(model_id=MODEL_ID), tools=[search_policy],
               system_prompt=SYSTEM)


@app.entrypoint
def invoke(payload):
    return {"result": str(_agent(payload.get("prompt", "Hello")))}


if __name__ == "__main__":
    app.run()
