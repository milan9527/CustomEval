"""Run a Claude Agent SDK query with OTEL export to the local collector.

Per code.claude.com/docs/en/agent-sdk/observability: CLAUDE_CODE_ENABLE_TELEMETRY=1
+ OTLP exporters + OTEL_LOG_RAW_API_BODIES=1 makes the CLI emit claude_code.*
logs (incl. api_request_body / api_response_body carrying the Messages-API bodies)
to the collector, which forwards them to CloudWatch Logs (/aws/saes/agentsdk-results).
"""
import asyncio
import os

from claude_agent_sdk import ClaudeAgentOptions, query

OTEL = {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",     # traces (beta)
    "OTEL_TRACES_EXPORTER": "otlp",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
    # short intervals so a short-lived run actually flushes (doc: "Flush telemetry")
    "OTEL_METRIC_EXPORT_INTERVAL": "1000",
    "OTEL_LOGS_EXPORT_INTERVAL": "1000",
    "OTEL_TRACES_EXPORT_INTERVAL": "1000",
    # emit the full Anthropic Messages-API request/response bodies as log events
    "OTEL_LOG_RAW_API_BODIES": "1",
    "OTEL_LOG_USER_PROMPTS": "1",
    "OTEL_LOG_TOOL_DETAILS": "1",
    "OTEL_LOG_TOOL_CONTENT": "1",
    "OTEL_SERVICE_NAME": "saes-agentsdk-demo",
    # run against Bedrock, same as the host CLI
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "AWS_REGION": "us-east-1",
    "ANTHROPIC_MODEL": os.environ.get("ANTHROPIC_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
    # keep the child from picking up this outer Claude Code session
    "CLAUDECODE": "",
    "CLAUDE_CODE_SESSION_ID": "",
    "CLAUDE_CODE_CHILD_SESSION": "",
}


async def main():
    env = {**os.environ, **OTEL}
    opts = ClaudeAgentOptions(
        env=env,
        system_prompt="You are a helpful assistant. Answer directly and briefly.",
        allowed_tools=["Bash"],
        max_turns=4,
    )
    prompt = "Run the bash command `echo hello-from-agent` and tell me its output."
    print(f"PROMPT: {prompt}")
    async for msg in query(prompt=prompt, options=opts):
        # print assistant text as it streams
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for b in content:
                t = getattr(b, "text", None)
                if t:
                    print("ASSISTANT:", t)
    print("done — telemetry flushing to collector")
    await asyncio.sleep(3)  # let the 1s export intervals flush


if __name__ == "__main__":
    asyncio.run(main())
