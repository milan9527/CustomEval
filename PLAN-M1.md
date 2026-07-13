# SAES — M1 Implementation Plan (Core, offline)

> **STATUS: M1 COMPLETE** ✅ — 78 tests passing, ruff clean, CLI smoke-tested end-to-end (`init` → `doctor` → `run` producing console summary + JSON + HTML + gate exit code).
> Implemented: T1 config, T2 judge (openai_compatible + bedrock), T3 ingest (thin `otlp_file` over native mappers), T4 evaluator resolution (13 native builtins + custom LLM via `OutputEvaluator` + custom code via `@code_evaluator`), T5 on-demand runner + gate (native `Experiment`), T6 JSON/HTML reports, T7 CLI.
> Carried to M2 (see PLAN-M2.md): realistic OpenInference/CloudWatch fixtures (§ "OTLP dialect drift"); finer-grained `saes doctor` field-coverage report.
>
> ⚠️ **Historical note — the T3/T4 task bodies below are superseded.** They describe an early plan to *port AgentCore prompt templates* and build a `transcript.py` + generic template `runner.py`. That approach was **reversed** once the native `strands-agents-evals` package was verified to already ship all evaluators, templates, and mappers. As shipped, SAES resolves ids to **native** evaluator classes and delegates ingestion to **native** mappers — there is no `transcript.py`, no `evaluators/runner.py`, no `evaluators/templates/`. See SPEC §4 ("Why native, not ported templates") and DOCUMENTATION.md §3 for the actual design. The task bodies are retained as a record of the decision path.

**Scope (from SPEC §14, M1):** Config schema → judge layer (`openai_compatible` + `bedrock`) → thin `otlp_file` source over native mappers → evaluator resolution (13 built-ins → native classes; custom LLM/code) → on-demand runner + CI gate → local JSON/HTML report.

**Guiding decisions (SPEC §15):**
- **D1 reuse (maximal):** ALL evaluators, prompt templates, session mappers, trace providers, and the `Experiment`/`Case`/`Report` engine come from native `strands-agents-evals`. Verified against the installed package (v1.0.2): the 13 built-ins exist as classes and accept `model=`; templates live in `strands_evals.evaluators.prompt_templates`; mappers/providers exist as documented. SAES reimplements **none** of this. An earlier draft that hand-ported templates + built a normalizer/runner was **removed** as duplicate work.
- **D2 framework-agnostic:** input contract is OTEL GenAI-convention spans. M1 proves this with `otlp_file` (no AWS): a non-Strands OpenInference dump maps to a native `Session` via `detect_otel_mapper`. (`cloudwatch` via native `CloudWatchProvider` is M2.)
- **Custom evaluators (SPEC §6):** parity with AgentCore — custom LLM (native `OutputEvaluator`) + custom code (native `Evaluator` subclass via `@code_evaluator`).
- **Score comparability:** built-ins are the native evaluators (AgentCore-equivalent prompts, versioned), so scores line up with managed AgentCore; hold the judge model constant.

**M1 exit criteria (definition of done):**
1. `saes run --config eval.yaml --data-source dump.otlp.jsonl` scores a third-party OTLP dump using an OpenAI-compatible judge and writes `results.json` + `report.html`.
2. Swapping the same run to `judge.provider: bedrock` works with no other change.
3. `saes doctor` reports OTEL-conformance coverage on that dump.
4. CI gate exits non-zero when a threshold fails.
5. No agent code is imported by SAES during a `otlp_file` run (framework-agnostic proof).

---

## 0. Repo bootstrap

```
saes/
├─ pyproject.toml            # deps: strands-agents, strands-agents-evals,
│                            #       strands-agents[openai], boto3, pydantic, jinja2, typer
├─ src/saes/
│  ├─ config/                # T1
│  ├─ judge/                 # T2
│  ├─ ingest/                # T3  (adapters + normalizer + profiles)
│  ├─ evaluators/            # T4  (wrappers over strands-agents-evals)
│  ├─ run/                   # T5  (on-demand runner + gate)
│  ├─ report/                # T6  (json + html)
│  └─ cli.py                 # T7  (typer: run | doctor | init)
└─ tests/
   ├─ fixtures/*.otlp.jsonl  # incl. a non-Strands / OpenInference dump
   └─ ...
```

Set up `ruff` + `pytest` + a `make check` target early. Python 3.10+.

---

## Task breakdown (in dependency order)

### T1 — Config schema & loader
**Files:** `config/schema.py`, `config/loader.py`
- Pydantic models: `EvaluationConfig`, `DataSourceConfig`, `JudgeModelConfig`, `EvaluatorRef`, `GroundTruthRef`, `ResultsSinkConfig`, `GateRule`. Mirror SPEC §12 YAML exactly.
- Secret handling: `api_key_env` resolves from env at load; never store the literal. Redact `base_url`/key in any serialization.
- `structured_output` enum: `json_schema | tool_call | prompt`.
- Validation: reject unknown built-in evaluator IDs; warn if a ground-truth-only evaluator is selected without a `groundTruth` source.
**Test:** round-trip the SPEC §12 example; assert secrets never serialize.

### T2 — Judge model layer  ⟵ *the differentiator*
**Files:** `judge/base.py`, `judge/providers.py`, `judge/structured.py`
- `resolve_judge(cfg) -> Judge` factory:
  - `openai_compatible` → `strands.models.openai.OpenAIModel(client_args={"api_key": <resolved>, "base_url": cfg.base_url}, model_id=cfg.model, params=...)`.
  - `bedrock` → `strands.models.BedrockModel(...)`.
  - `strands` → named provider lookup (defer full registry to later; stub with clear error).
- `Judge.score(prompt, schema) -> Verdict{reason, score, label, rawScore, tokenUsage}`.
- `structured.py`: negotiate output enforcement in priority order `json_schema → tool_call → prompt`; JSON extractor + **one repair retry**; after `max_retries` mark `errored` (retain raw response, never silently drop).
- Reasoning-before-score contract enforced by schema field order.
- Per-evaluator `judge_override` supported by re-resolving with a merged config.
**Test:** mock an OpenAI-compatible server (respond with/without `response_format` support) → assert fallback path and repair retry. Assert `judgeModel` provenance stamped.

### T3 — Ingestion — **reuse `strands-agents-evals` providers + mappers** (verified, scope shrunk)
**Files:** `ingest/source.py`, `ingest/otlp_file.py`, `ingest/transcript.py`
- **Verified (R2):** the SDK already ships what T3 was going to build. Do not reimplement.
  - Providers (`get_evaluation_data(session_id) -> {output, trajectory: Session}`): `CloudWatchProvider` (reads AgentCore runtime log groups, discovers by `agent_name`), `LangfuseProvider`, `OpenSearchProvider`.
  - Session mappers (the "attribute-mapping profile" role): `CloudWatchSessionMapper`, `OpenInferenceSessionMapper`, `LangChainOtelSessionMapper`, `StrandsInMemorySessionMapper`, plus `detect_otel_mapper(spans)` auto-detection.
- `source.py`: thin factory `DataSourceConfig -> TraceProvider`/mapper selection. For M1: `otlp_file` (local) + pass-through to the SDK's `live` capture. (`CloudWatchProvider` wiring is trivially available but formally an M2 deliverable.)
- `otlp_file.py`: read `.jsonl`/OTLP dump → span dicts → `detect_otel_mapper(spans).map_to_session(...)` → SDK `Session`. **No AWS.** M1 workhorse + D2 proof. The **non-Strands/OpenInference** dump is handled by the SDK's `OpenInferenceSessionMapper` — we test it, we don't build it.
- `transcript.py` — **the one genuinely net-new piece (small):** format the SDK `Session` into AgentCore's `{context}` transcript shape (`User:/Assistant:/Action:/Tool:`) + `{assistant_turn}`/`{available_tools}`/`{tool_turn}` placeholders. Needed only because we use AgentCore's verbatim prompts (T4). Golden-tested against a known transcript.
**Test:** feed a non-Strands/OpenInference fixture dump through `detect_otel_mapper` → assert a correctly reconstructed `Session` with tool trajectory (this exercises the SDK path = D2 guarantee). Golden-test `transcript.py` output.

### T4 — Built-in evaluators (5, trace-level) — **port AgentCore templates + one generic runner**
**Files:** `evaluators/registry.py`, `evaluators/runner.py`, `evaluators/templates/*.yaml`, `evaluators/templates/*.txt`
- M1 set: `Helpfulness`, `Correctness`, `Coherence`, `ResponseRelevance`, `Faithfulness` (all trace-level; `Correctness` gains its `expectedResponse` ground-truth variant in M2).
- **Not wrappers.** Port the AgentCore-published prompt templates (SPEC §4). Each evaluator = a `.yaml` metadata file (id, level, placeholders, score-scale map, verdict field, `source`, `template_ref`) + the `.txt` prompt.
- `runner.py` — **one generic runner drives all templates**, implemented as a single `Evaluator` subclass (verified R1: `Evaluator.evaluate_async` returning `list[EvaluationOutput]`, judge injected via `model=`): gather placeholder values via `transcript.py` (T3) per level → render template → call the injected OpenAI-compatible judge with the template's JSON schema (`structured_output_model`) → parse verdict enum → map to normalized `[0,1]` (preserve raw enum in `rawScore`). Adding an evaluator is a data change (new `.yaml`+`.txt`), not a new class.
- Score-scale maps captured verbatim from the published templates: Helpfulness/Coherence/Faithfulness = 5-level ordinal → 0/.25/.5/.75/1; Correctness = `Perfectly/Partially/Incorrect` → 1/.5/0; (M2 adds Harmfulness/Refusal binary + `higher_is_better` flag).
- `registry.py`: resolve `"Builtin.Helpfulness"` → template instance; hook for custom evaluators later.
- **Engine reuse (D1):** the runner plugs into `strands-agents-evals` `Experiment`/`Case`/`Report` orchestration; only the prompt-content layer is AgentCore's. Trajectory scorers (M2) come straight from `strands_evals`.
- Normalize output to SAES `EvaluationResult` (score→[0,1], `rawScore`, `reason`, `label`, `judgeModel`, template `source`+version, `groundTruthUsed`, `ignoredReferenceInputFields`).
**Test:** run each template against a fixture trace with a stubbed judge returning each enum value → assert correct normalized mapping and provenance stamping. Golden-compare rendered prompt against the published template for one evaluator.

### T5 — On-demand runner + gate
**Files:** `run/runner.py`, `run/gate.py`
- `runner.run(config)`: load data source → normalize → for each unit × evaluator, call the wrapper (bounded concurrency; reuse judge client). Aggregate per-evaluator stats (avg, p50, pass-rate, errored count).
- Ground truth: join dataset (`sessionId`/`traceId`) onto views; each evaluator reads only its field; record `ignoredReferenceInputFields`.
- `gate.py`: parse rules like `"Builtin.Helpfulness.avg >= 0.8"`; return non-zero exit on failure; errored results excluded from aggregates but surfaced in summary counts.
- Existing-session runner (score by ID) can be stubbed in M1 (full support M2 with `cloudwatch`).
**Test:** end-to-end on `otlp_file` fixture + stub judge → assert aggregates and gate pass/fail exit codes.

### T6 — Local results sink (JSON + HTML)
**Files:** `report/json_sink.py`, `report/html_report.py`, `report/templates/report.html.j2`
- JSON: array of `EvaluationResult` + a run summary (config hash, judge model, counts, aggregates).
- HTML (Jinja2, self-contained, CI-artifact friendly): per-evaluator score summary, per-trace drill-down with judge reasoning, errored-result list. No external assets.
**Test:** snapshot the JSON schema; assert HTML renders with N results and shows reasoning.

### T7 — CLI
**Files:** `cli.py` (typer)
- `saes run --config <yaml> [--data-source <path>] [--dataset <jsonl>] [--gate ...]` → T5 + T6, exit code from gate.
- `saes doctor --data-source <path>` → sample spans, report per-field OTEL-conformance coverage (SPEC §7.1a). Surfaces exactly which `gen_ai.*` attrs / grouping keys are missing.
- `saes init` → interactive scaffold: pick agent type (customer-service/RAG/tool-heavy) → emit a starter `eval.yaml` with the recommended 3–4 evaluators and a judge stub.
**Test:** CLI smoke tests via `typer.testing.CliRunner`; assert `run` exit codes and `doctor` output.

---

## Sequencing & parallelism

```
T1 (config) ──┬─> T2 (judge) ─────────┐
              ├─> T3 (ingest) ─────────┼─> T5 (runner+gate) ─> T6 (report) ─> T7 (CLI)
              └─> (fixtures)           │
                    T4 (evaluators) <──┘  (needs T2 judge + T3 views)
```
- **T1 first** (everything depends on config types).
- **T2 and T3 in parallel** (independent: judge vs. ingestion).
- **T4 after T2+T3** (needs judge + views).
- **T5 → T6 → T7** sequential.
- Author OTLP fixtures (incl. one non-Strands/OpenInference dump) alongside T3 — they gate the D2 exit criterion.

Two contributors: one takes T2+T4 (judge path), one takes T3+fixtures (ingestion path); converge on T5.

---

## Risks / watch-items for M1

**Pre-implementation verification (done) — the two schedule risks are resolved against the real `strands-agents-evals` API:**

- **R1 — judge injection & orchestration → RESOLVED.** The SDK's own custom-`Evaluator` example injects the judge via a plain `model=` arg (`Agent(model=..., ...)(prompt, structured_output_model=EvaluationOutput)`). Our OpenAI-compatible `OpenAIModel` drops straight in — no adaptation. Base class = implement `evaluate/evaluate_async → list[EvaluationOutput]{score,test_pass,reason,label}`; orchestration = `Experiment(cases, evaluators).run_evaluations_async(task)` → `report.overall_score/.reasons/.to_file`. T4/T5 build directly on these.
- **R2 — framework-agnostic ingestion → RESOLVED (and scope shrunk).** Providers (`CloudWatchProvider`/`Langfuse`/`OpenSearch`) and mappers (`CloudWatch`/`OpenInference`/`LangChainOtel`/`StrandsInMemory` + `detect_otel_mapper`) already exist. The non-Strands/OpenInference path is a *tested* SDK capability, not net-new code. T3 shrinks to a thin source factory + fixtures.

**Remaining watch-items (real, but not blockers):**

- **Transcript formatting is the one net-new coupling** — AgentCore's verbatim prompts need the `User:/Assistant:/Action:/Tool:` `{context}` shape; `transcript.py` (T3) produces it from the SDK `Session`. Golden-test it, or scores drift from managed AgentCore.
- **Structured output on arbitrary endpoints** — vLLM/Ollama guided-decoding support varies; T2's fallback + repair path must be solid, and `saes doctor` should probe it. (SDK uses `structured_output_model`; confirm it degrades gracefully on prompt-only endpoints.)
- **OTLP dialect drift** — use a *real* OpenInference fixture, not hand-idealized, so the mapper path holds up for M2's CloudWatch adapter.
- **Score comparability** — stamp `judgeModel` + template version on every result now (cheap), so M2 baselines are meaningful.

---

## What M1 deliberately defers

- `cloudwatch` adapter, EMF/log results sink, dashboards → **M2**
- Full 13 evaluators + trajectory scorers + full ground-truth dataset support → **M2**
- Online sampling worker, custom LLM/code evaluators → **M3**
- Experiment generation / simulators / detectors / Langfuse → **M4**
