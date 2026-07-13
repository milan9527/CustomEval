# SAES — M3 Implementation Plan (Online evaluation)

> **STATUS: M3 COMPLETE** ✅ — T14 (session-completion tracker), T15 (worker loop + `saes serve`), T16 (Lambda code evaluator), T17 (dashboard/alarms CDK) — **168 tests + 4 CDK synth tests passing, ruff clean**. The online path was also **verified end-to-end against a real deployed AgentCore Runtime agent** (see DOCUMENTATION.md §10, F6).
>
> **Verified during build:** the native `CloudWatchProvider` reads only by known session id, so SAES owns `discover_sessions_with_last_seen` (Logs Insights `stats max(@timestamp) by session.id`; handles numeric-ms and ISO timestamp forms). Rate cap is a rolling 60s window across cycles (not per-cycle). Failed scores are NOT marked, so they retry next cycle; only successes are marked (injected clock — no real sleeping in tests). `saes serve --once` runs one cycle; the full loop sleeps `--interval`.
> **T16:** the same `@code_evaluator` function body runs locally (`CodeEvaluator`) and as a Lambda (`online/lambda_evaluator.py` `handle`), mapping the event → `EvaluationData` → `CodeVerdict` → AgentCore-style `{score,label,reason}` response.
> **T17:** `cdk/` (separate `saes-cdk` package) synthesizes a dashboard + per-evaluator alarms + a least-privilege worker IAM role over `SAES/Evaluations`. Validated via `cdk synth` and CDK `assertions` tests (CLI/lib schema must match — pinned in `cdk/requirements.txt`).

**Scope (from SPEC §8.2, §14):** A long-running worker that continuously monitors production traffic — detects completed sessions, samples them, scores with the configured evaluators + judge, and writes results to CloudWatch (reusing the T9 EMF/JSON sink). Plus custom code evaluators deployed as Lambda, and optional dashboard/alarm CDK.

**Design anchor — mirror managed AgentCore Evaluations (researched, not assumed):** managed AgentCore reads spans from CloudWatch, groups them into whole-conversation sessions, and on a recurring schedule samples + scores completed sessions, writing EMF metrics to `Bedrock-AgentCore/Evaluations` and JSON results to a dedicated log group under a separate execution IAM role. SAES's online worker replicates this shape with the OpenAI-compatible judge and `SAES/Evaluations` namespace. M1/M2 already built the pieces the worker composes: session discovery (T8), evaluator resolution + judge (T4/T2), native `Experiment` scoring (T5), EMF/JSON sink (T9).

**Guiding decisions (unchanged):** D1 reuse, D2 framework-agnostic. The worker adds only the scheduling/state loop around the existing pipeline.

**M3 exit criteria:**
1. `saes serve --config online.yaml` runs a polling loop that scores newly-completed production sessions and emits results to CloudWatch, never re-scoring a session.
2. Session-completion is detected by **span-quiescence timeout** (SPEC §8.2), verified with a simulated clock (no real waiting).
3. Sampling percentage + `max_per_minute` caps are enforced; any drop is logged, never silent.
4. A custom code evaluator can be deployed as a Lambda and invoked by the worker (parity with AgentCore code-based evaluators).
5. Worker is packageable as a container and as a scheduled Lambda.

---

## Task breakdown

### T14 — Session-completion tracker (span-quiescence)  ⟵ *the core new mechanism*
**Files:** `online/session_tracker.py`
- Track, across polling cycles, each session id → latest span timestamp (from the discovery query, extended to also return max @timestamp per session).
- A session is **complete** when `now - last_span_ts >= session.timeout_minutes` AND it hasn't already been scored.
- Persist "already scored" + "last seen" state so restarts don't re-score or lose progress. M3 start: in-memory + optional JSON state file; DynamoDB/parameter-store backend is a later option (log the choice).
- **Verify against the SDK/CloudWatch first:** the T8 discovery query currently returns session ids only. Extend it to `stats max(@timestamp) as last_ts by sid` (or similar) — confirm the Logs Insights field for span time before building.
**Test:** feed synthetic (session_id, last_ts) sets with an injected `now`; assert only quiescent, unscored sessions are returned; assert scored sessions never reappear; assert in-progress sessions are skipped until quiescent.

### T15 — Online worker loop
**Files:** `online/worker.py`, `cli.py` (`serve` command)
- Loop: discover sessions + last-ts (T8 extended) → T14 selects completed+unscored → apply sampling (`percentage`, `filters`) and `max_per_minute` cap → score each via the existing runner path (`_case_for` + native `Experiment` or per-session `get_evaluation_data`) → emit via T9 sink → mark scored.
- Poll interval configurable; graceful shutdown; structured progress logging.
- Cap enforcement: if more sessions are eligible than `max_per_minute`, defer the remainder to the next cycle and **log the deferral count** (never silently drop).
- Reuse — the worker orchestrates M1/M2 components; it must not reimplement scoring, discovery, or emission.
**Test:** one full cycle with stubbed discovery + stub judge + capturing sink; assert selected sessions scored once, results emitted, caps respected, deferrals logged. Use an injected clock; no real sleeping.

### T16 — Custom code evaluator as Lambda (AgentCore parity)
**Files:** `online/lambda_evaluator.py`, packaging notes
- Deploy a SAES `@code_evaluator` function as a Lambda; the worker invokes it with the session's spans as a structured event and expects `{score, label, reason}` — matching AgentCore's code-based evaluator contract so the same function body works in both.
- A thin Lambda handler that wraps the registered callable and adapts event → `EvaluationData` → `CodeVerdict` → response.
**Test:** invoke the handler locally with a sample event; assert the verdict shape. (Real Lambda deploy behind an integration flag.)

### T17 — Dashboard & alarms CDK (optional)
**Files:** `cdk/` (separate `saes-cdk` package per SPEC §11.1)
- CloudWatch dashboard widgets over `SAES/Evaluations` metrics (score trends per evaluator) + alarms (e.g. `Helpfulness.avg < 0.75`).
- IAM: least-privilege execution role (`PutLogEvents`, `PutMetricData`, `logs:StartQuery`/`GetQueryResults`, `lambda:InvokeFunction` for code evaluators) — separate from the agent's runtime role, mirroring AgentCore.
**Test:** CDK synth assertion (no deploy).

---

## Sequencing

```
T14 (session tracker) ──> T15 (worker loop) ──> T17 (dashboard/alarms)
T16 (lambda evaluator) — parallel, folds into T15's scoring step
```
- **T14 first** — the worker can't be correct without completion detection. Verify the Logs Insights timestamp field before coding.
- **T15** composes T14 + existing M1/M2 pipeline.
- **T16** parallel; **T17** last.

---

## Risks / verify against the real SDK before building

- **Logs Insights last-span timestamp** — confirm the exact field/aggregation to get per-session max span time (extend the T8 discovery query). Don't assume `@timestamp` semantics; test against a real (or documented) query.
- **State durability** — in-memory scored-set is lost on restart → potential re-scoring (wasted judge spend) or gaps. Decide the M3 default (JSON state file) and document the durable option.
- **Clock in tests** — never `sleep` in tests; inject `now`. (`Date.now()`-style calls are also banned in workflow scripts, same discipline.)
- **Cap accounting across cycles** — `max_per_minute` must be enforced over wall-clock, not per-cycle, or a fast poll interval defeats it. Track a rolling window.

---

## Deferred beyond M3 (unchanged)

- Experiment generation / simulators / detectors → **M4**
- Langfuse / OpenSearch sources → **M4**
- A/B testing, batch optimization integration → future
