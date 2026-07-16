"""Local smoke test: run one invocation with an in-memory OTEL SDK, write the
spans + events as a CloudWatch-shaped JSONL dump, then check it with
`saes doctor --data-source` before deploying. Not part of the deployed image."""
import asyncio
import json
import sys

from opentelemetry import _events as events_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

span_exporter = InMemorySpanExporter()
tp = TracerProvider()
tp.add_span_processor(SimpleSpanProcessor(span_exporter))
trace_api.set_tracer_provider(tp)

log_exporter = InMemoryLogExporter()
lp = LoggerProvider()
lp.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
events_api.set_event_logger_provider(EventLoggerProvider(lp))

import agent  # noqa: E402  (providers must be set before the module imports)


class Ctx:
    session_id = "smoke-session-0001"


async def main():
    out = await agent.invoke({"prompt": sys.argv[1] if len(sys.argv) > 1 else
                              "What is the weather in Tokyo?"}, Ctx())
    print("ANSWER:", out["result"])

    records = []
    for s in span_exporter.get_finished_spans():
        records.append({
            "traceId": format(s.context.trace_id, "032x"),
            "spanId": format(s.context.span_id, "016x"),
            "scope": {"name": s.instrumentation_scope.name},
            "attributes": dict(s.attributes or {}),
        })
    for ld in log_exporter.get_finished_logs():
        r = ld.log_record
        records.append({
            "traceId": format(r.trace_id or 0, "032x"),
            "spanId": format(r.span_id or 0, "016x"),
            "scope": {"name": "claudesdk.agent"},
            "attributes": dict(r.attributes or {}),
            "body": r.body,
        })
    with open("smoke_dump.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records to smoke_dump.jsonl")


asyncio.run(main())
