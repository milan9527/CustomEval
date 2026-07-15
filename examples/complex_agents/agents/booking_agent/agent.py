"""Travel-booking agentic-workflow agent on AgentCore (Strands, native OTEL).

Multi-tool orchestration: search flights → search hotels → book → confirm. Tests
tool selection/parameter accuracy, trajectory, and goal completion. Deterministic
backend. Policy: search before booking; never book without explicit confirmation."""
import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel


@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search flights. Returns options with fare ids and prices (USD)."""
    return str([
        {"fare_id": "FL-AA100", "airline": "AA", "depart": f"{date} 08:00", "price": 320},
        {"fare_id": "FL-DL220", "airline": "DL", "depart": f"{date} 14:30", "price": 285},
    ])


@tool
def search_hotels(city: str, checkin: str, nights: int) -> str:
    """Search hotels in a city. Returns options with rate ids and nightly prices."""
    return str([
        {"rate_id": "HT-GRAND", "hotel": "Grand Central", "per_night": 180},
        {"rate_id": "HT-BUDGET", "hotel": "City Inn", "per_night": 95},
    ])


@tool
def book_flight(fare_id: str, passenger: str) -> str:
    """Book a flight by fare id. Only after the user confirms the choice."""
    return f"booked {fare_id} for {passenger} (PNR {fare_id[-3:]}2Z)"


@tool
def book_hotel(rate_id: str, guest: str, nights: int) -> str:
    """Book a hotel by rate id. Only after the user confirms the choice."""
    return f"booked {rate_id} x{nights} nights for {guest} (conf {rate_id[-3:]}9Q)"


SYSTEM = (
    "You are a travel-booking assistant. Always search before booking. Present "
    "options with prices and let the user choose; never book until the user confirms "
    "a specific option. Use the exact fare_id/rate_id from search results. Confirm "
    "each booking with its reference code. Keep the user's stated constraints "
    "(dates, budget, city) accurate across turns."
)

app = BedrockAgentCoreApp()
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
_agent = Agent(model=BedrockModel(model_id=MODEL_ID),
               tools=[search_flights, search_hotels, book_flight, book_hotel],
               system_prompt=SYSTEM)


@app.entrypoint
def invoke(payload):
    return {"result": str(_agent(payload.get("prompt", "Hello")))}


if __name__ == "__main__":
    app.run()
