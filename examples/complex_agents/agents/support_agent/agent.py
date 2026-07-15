"""Customer-support / helpdesk agent on AgentCore (Strands, native OTEL→CloudWatch).

Realistic SaaS support desk: looks up an account, checks subscription + PTO-style
balances, files a refund, and escalates. Multi-tool, multi-turn. The system prompt
enforces real support policy (verify identity, cite the plan, escalate >$100)."""
import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel

# --- fake backend (deterministic so evaluations are reproducible) ------------
_ACCOUNTS = {
    "A-1001": {"name": "Dana Lee", "plan": "Pro", "seats": 5, "mrr": 90,
               "status": "active", "renews": "2026-09-01"},
    "A-2002": {"name": "Sam Ortiz", "plan": "Starter", "seats": 1, "mrr": 15,
               "status": "past_due", "renews": "2026-07-20"},
}
_INVOICES = {"A-1001": [{"id": "INV-77", "amount": 90, "refundable": True}],
             "A-2002": [{"id": "INV-31", "amount": 15, "refundable": False}]}


@tool
def lookup_account(account_id: str) -> str:
    """Look up a customer account by id (e.g. 'A-1001'). Returns plan + status."""
    a = _ACCOUNTS.get(account_id.strip().upper())
    return str(a) if a else f"no account {account_id}"


@tool
def get_invoices(account_id: str) -> str:
    """List recent invoices for an account, with refund eligibility."""
    return str(_INVOICES.get(account_id.strip().upper(), []))


@tool
def issue_refund(invoice_id: str, amount: float) -> str:
    """Issue a refund for an invoice. Only call after confirming eligibility."""
    return f"refund of ${amount:.2f} issued for {invoice_id} (ref RFND-{invoice_id[-2:]})"


@tool
def escalate(account_id: str, reason: str) -> str:
    """Escalate to a human agent (use for refunds > $100 or angry customers)."""
    return f"escalated {account_id} to tier-2 (ticket ESC-{account_id[-4:]}): {reason}"


SYSTEM = (
    "You are a helpful SaaS customer-support agent. Policy: (1) always look up the "
    "account before answering account-specific questions; (2) state the plan and "
    "status you found; (3) only issue a refund after get_invoices confirms it is "
    "refundable; (4) escalate refunds over $100 or unresolved billing. Be concise "
    "and cite concrete values from the tools."
)

app = BedrockAgentCoreApp()
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
_agent = Agent(model=BedrockModel(model_id=MODEL_ID),
               tools=[lookup_account, get_invoices, issue_refund, escalate],
               system_prompt=SYSTEM)


@app.entrypoint
def invoke(payload):
    return {"result": str(_agent(payload.get("prompt", "Hello")))}


if __name__ == "__main__":
    app.run()
