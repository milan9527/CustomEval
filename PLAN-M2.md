# SAES — M2 Implementation Plan (Observability integration)

> **STATUS: M2 COMPLETE** ✅ — T8 (CloudWatch source + SAES-owned session discovery), T9 (EMF/JSON results sink), T10 (all 13 built-ins verified e2e), T11 (realistic fixture, non-empty reconstruction), T12 (`saes doctor` field-coverage), T13 (trajectory scorers) — **129 tests passing, ruff clean**.
>
> **T13 resolved:** the three trajectory matchers (`exact_match_scorer`/`in_order_match_scorer`/`any_order_match_scorer`) are native SDK scorer *functions* (called via `.__wrapped__` to bypass the `@tool` decorator), not standalone evaluators. SAES wraps them as `TrajectoryMatchEvaluator` (native `Evaluator` subclass), selectable as `Builtin.Trajectory{Exact,InOrder,AnyOrder}Match` — deterministic, no LLM. Actual tool names extracted from `Session` (`ToolExecutionSpan.tool_call.name`), expected from `Case.expected_trajectory`. Verified matcher math (0.5/1.0/1.0/0.0 cases). SPEC §4.2 updated to match reality.
>
> **T12 earned its keep immediately:** `saes doctor` on `otel_session.jsonl` showed `scope name (mapper selection): 0/3 ✗` — no `scope.name`, so `detect_otel_mapper` routed it to the Strands in-memory mapper, which failed on dict spans. Precisely diagnosed the M1 fixture gap; `doctor` reports it cleanly (exit 2, no traceback).
>
> **T11 closed the gap:** by reading the native OpenInference mapper source, the real required format is live-instrumentation attributes (`llm.input_messages.{i}.message.role`/`.content`, `llm.output_messages...`, `scope.name = openinference.instrumentation.langchain`). `openinference_real.jsonl` now reconstructs a **non-empty** trace with a real `InferenceSpan` — the D2 guarantee is upgraded from "returns a Session" to "returns a faithful Session." (The older minimal `openinference_session.jsonl` is kept for the stub-mapper seam tests.)
>
> **Verified findings during build (real SDK v1.0.2):**
> - `CloudWatchProvider` reads **only by known `session_id`** (no enumeration API). → SAES owns a Logs Insights **session-discovery** query (`ingest/cloudwatch.py`); per-session read/map is delegated to the provider + its `as_task()`. Confirmed, built, tested with a stubbed client.
> - **Trajectory scoring is NOT a standalone deterministic evaluator.** Native `TrajectoryEvaluator(rubric=..., model=...)` is LLM-based; the exact/in-order/any-order matchers are *scorer tools* the judge calls internally, not directly-selectable evaluators. `deterministic.trajectory` exposes `ToolCalled`, not the three matchers as evaluators. → SPEC §4.2 "trajectory scorers" needs remapping: either (a) expose `expectedTrajectory` via `TrajectoryEvaluator` (LLM+scorer-tool), or (b) wrap the matcher functions as SAES code evaluators. **Decide in T13 below; do not fake standalone matchers.**

**Scope (from SPEC §14, M2):** CloudWatch trace source (native `CloudWatchProvider`) → EMF metrics + JSON-log results sink → full 13 built-in evaluators verified → ground-truth dataset support hardened → CI gate polish. Builds directly on the M1 core (T1–T7 complete).

**Guiding decisions (unchanged):** D1 reuse (native `strands-agents-evals` providers/mappers/evaluators), D2 framework-agnostic (OTEL contract). M2 adds the AWS-facing surface M1 deferred.

**Verified against installed `strands-agents-evals` v1.0.2:**
- `CloudWatchProvider(region, log_group, agent_name, lookback_days, query_timeout_seconds, mapper, end_time)` exists; `get_evaluation_data(session_id) -> TaskOutput`; and `provider.as_task()` returns the exact `task(case)` closure the native `Experiment` needs. **This means the runner's task can come straight from the provider** — no SAES glue for the CloudWatch read path.
- `boto3` 1.43 available for the write path (`put_metric_data`, `put_log_events` / EMF).

**M2 exit criteria (definition of done):**
1. `saes run` with `dataSource.type: cloudwatch` scores real sessions read from a CloudWatch log group via the native provider (no local dump).
2. Results land in CloudWatch: per-evaluator metrics (EMF) under a namespace + full JSON result records in a log group.
3. All 13 built-in ids resolve and run end-to-end (extends M1's smoke coverage from 1 evaluator to all 13).
4. Ground-truth dataset flows into `expected_*` for `Correctness`/`GoalSuccessRate`/trajectory scorers, with a realistic fixture.
5. A **real** OpenInference (and CloudWatch-format) fixture reconstructs non-empty sessions — closing the M1 fixture-fidelity gap.
6. `saes doctor` reports per-field OTEL-conformance coverage, not just session count.

---

## Task breakdown

### T8 — CloudWatch trace source
**Files:** `ingest/source.py` (extend), `ingest/cloudwatch.py`
- Wire `DataSourceConfig.type == "cloudwatch"` to the native `CloudWatchProvider`, built from the config's `cloudwatch` block (`log_group_names`/`agent_name`/`region`/`lookback_days`).
- **Reuse `provider.as_task()`** for the runner's task instead of SAES's `_final_output`/closure. Refactor `run_on_demand` so the task source is pluggable: local dump → in-memory sessions + closure (M1 path); cloudwatch → `provider.as_task()` + a case list built from discovered session ids.
- Session discovery: provider reads by `session_id`; need the set of session ids in the window. Check whether the provider exposes a list/scan; if not, document that cloudwatch runs require an explicit session-id list (or a CloudWatch Logs Insights query in SAES) — **verify against the SDK before designing**, don't assume.
- `region` falls back to `AWS_REGION`/`AWS_DEFAULT_REGION` (provider already does this).
**Test:** unit-test source resolution with a stubbed provider (no AWS); assert `run_on_demand` uses `as_task()` for the cloudwatch path. Live AWS test gated behind an env flag / marked integration.

### T9 — Results sink to CloudWatch (EMF metrics + JSON logs)
**Files:** `report/cloudwatch_sink.py`, wire into `cli.run`
- **Metrics (EMF):** per-evaluator `avg`/`pass_rate`/`errored` as CloudWatch custom metrics under `metrics_namespace` (config `resultsSink.cloudwatch`), dimensioned by `agentId`/`evaluatorId`/env. Emit via EMF-structured `put_log_events` or `put_metric_data` — pick one, justify (EMF preferred: metrics + structured logs in one call, renders in the gen-AI observability views, SPEC §10.2).
- **Logs:** full `ReportDocument` rows as JSON records to `resultsSink.cloudwatch.log_group`.
- Secrets/creds: use the ambient boto3 session (IAM role); never inline keys. Respect least-privilege (`PutLogEvents`, `PutMetricData`).
- No-op gracefully when `resultsSink.cloudwatch` is absent (local-only run stays offline).
**Test:** stub the boto3 client (moto or a hand stub); assert the EMF payload shape (namespace, dimensions, metric names) and that JSON records carry judge reasoning. Assert no calls when sink unconfigured.

### T10 — All 13 built-ins verified end-to-end
**Files:** `tests/test_builtins_e2e.py`
- Extend M1's single-evaluator smoke to all 13 ids: resolve each, run through the native `Experiment` with a fixed-verdict stub model, assert each produces a scored row with correct evaluator name.
- Cover the ground-truth consumers (`Correctness` with `expectedResponse`, `GoalSuccessRate` with `assertions`) and the `ContextRelevance` alias flag.
- Trajectory scorers (native `TrajectoryEvaluator` / deterministic matchers): confirm they attach when `expectedTrajectory` present. **Verify the native trajectory API shape first.**
**Test:** parametrized over all 13 ids.

### T11 — Realistic fixtures (close M1 gap)
**Files:** `tests/fixtures/openinference_real.jsonl`, `tests/fixtures/cloudwatch_format.jsonl`
- Capture or hand-build fixtures that satisfy the native mappers' actual span-shape requirements (M1's minimal OpenInference fixture reconstructed 0 traces — "No user contents"/"Missing required fields"). Derive the required shape from the native mapper source, or capture from a real instrumented run.
- Update `test_ingest.py` to assert **non-empty** traces + a correct tool trajectory, upgrading the D2 guarantee from "returns a Session" to "returns a faithful Session."

### T12 — `saes doctor` field-coverage report
**Files:** `cli.py` (extend `doctor`)
- Beyond session count: sample spans and report presence/absence of the key GenAI attributes and grouping keys the mappers need (per SPEC §7.1a), so a third party sees exactly what their instrumentation is missing.
- Reuse the detected mapper's expectations where possible rather than hard-coding.

---

## Sequencing

```
T8 (cloudwatch source) ──┐
T11 (real fixtures) ─────┼─> T10 (13 builtins e2e, needs good fixtures)
T9 (results sink) ───────┘
T12 (doctor) — independent, any time
```
- **T11 first or parallel** — realistic fixtures unblock honest T10/T8 tests.
- **T8 and T9 parallel** (read path vs write path, independent).
- **T10** after fixtures. **T12** independent.

---

## Risks / things to verify against the real SDK before building

- **Session discovery for CloudWatch** — the native provider reads *by session_id*. How does SAES enumerate session ids in a time window? Options: SDK helper (verify), a SAES CloudWatch Logs Insights query, or require an explicit id list. **Resolve by inspecting the provider before T8 design.** Don't assume a list API exists.
- **EMF vs put_metric_data** — confirm which renders in the AgentCore gen-AI observability dashboard (SPEC §10.2 open item). Prefer EMF; validate the exact envelope.
- **Trajectory evaluator API** — confirm native `TrajectoryEvaluator`/scorer construction + how `expected_trajectory` on the Case reaches it, before T10.
- **Native mapper span-shape** — the required OpenInference/CloudWatch attributes must come from the mapper source or a real capture, not guessed (root cause of the M1 fixture gap).

---

## Deferred beyond M2 (unchanged)

- Online sampling worker (container + Lambda), dashboards/alarms CDK → **M3**
- Custom code evaluator as deployed Lambda → **M3**
- Experiment generation / simulators / detectors; Langfuse/OpenSearch sources → **M4**
