"""No-framework agent: plain Python + raw Bedrock call. Emits OTEL
traceloop-convention spans (root workflow + child LLM) to a local JSONL dump."""
import json, os, sys, boto3

def ask(prompt):
    br = boto3.client("bedrock-runtime", region_name="us-east-1")
    body = {"anthropic_version":"bedrock-2023-05-31","max_tokens":120,
            "messages":[{"role":"user","content":prompt}]}
    return json.loads(br.invoke_model(modelId=os.environ["BEDROCK_MODEL_ID"],
                      body=json.dumps(body))["body"].read())["content"][0]["text"]

def turn(sid, tid, q):
    a = ask(q)
    print(f"  [{sid}] Q: {q}\n         A: {a}", file=sys.stderr)
    sc = {"name":"opentelemetry.instrumentation.langchain"}
    return [
      {"span_id":f"{tid}-wf","trace_id":tid,"name":"agent.workflow","scope":sc,
       "attributes":{"session.id":sid,"traceloop.span.kind":"workflow","traceloop.entity.name":"MyAgent",
         "traceloop.entity.input":json.dumps({"inputs":{"messages":[{"role":"user","content":q}]}}),
         "traceloop.entity.output":json.dumps({"outputs":{"messages":[{"role":"assistant","content":a}]}})}},
      {"span_id":f"{tid}-llm","trace_id":tid,"parent_span_id":f"{tid}-wf","name":"chat","scope":sc,
       "attributes":{"session.id":sid,"llm.request.type":"chat",
         "gen_ai.prompt.0.role":"user","gen_ai.prompt.0.content":q,
         "gen_ai.completion.0.role":"assistant","gen_ai.completion.0.content":a}},
    ]

if __name__ == "__main__":
    spans=[]
    for sid,tid,q in [("s-geo","t1","What is the capital of France? One sentence."),
                      ("s-math","t2","What is 25 times 4?")]:
        spans += turn(sid,tid,q)
    open("traces.jsonl","w").write("\n".join(json.dumps(s) for s in spans))
    print(f"wrote {len(spans)} spans / 2 sessions -> traces.jsonl")
