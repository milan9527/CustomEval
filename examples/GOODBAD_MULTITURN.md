# Good vs. bad multi-turn agents across frameworks — full run + evaluation record

Goal: run **multi-turn** sessions against the deployed framework agents, including
a **deliberately bad** agent whose answers should drive metrics DOWN, then
evaluate everything with SAES and record both the **agent-run process** and the
**evaluation process** verbatim.

- Agents (all on AgentCore Runtime, auto-OTEL → CloudWatch):
  **good** = Strands / no-framework / LangGraph / CrewAI (the tool-calling agents);
  **bad** = a Strands agent with an evasive system prompt (`agents/bad_agent/`).
- Scenario: the same **3-turn** tool-calling conversation for every agent, all
  three turns sharing one AgentCore session id (`--session-id`) so SAES
  reconstructs a multi-turn Session.
- Judge: Amazon Bedrock OpenAI-compatible (`openai.gpt-oss-20b-1:0`).
- Reproduce: `bash goodbad_run.sh` (drive the agents) then
  `python goodbad_eval.py` (score). Raw artifacts: `GOODBAD_TRANSCRIPT.txt`
  (agent responses), `GOODBAD_EVAL_OUTPUT.txt` (scores + judge reasoning).

---

## Part 1 — The agent run (verbatim)

The 3 turns, sent to every agent:

1. `What is the weather in Tokyo?`
2. `What is 15% of 240?`
3. `Weather in Paris, and what is 12 times 8?`

Command per agent (from its `agents/<fw>/` dir):

```bash
sid="good-strands-<stamp>-...padding..."       # one id, reused for all 3 turns
agentcore invoke --session-id "$sid" '{"prompt": "What is the weather in Tokyo?"}'
agentcore invoke --session-id "$sid" '{"prompt": "What is 15% of 240?"}'
agentcore invoke --session-id "$sid" '{"prompt": "Weather in Paris, and what is 12 times 8?"}'
```

### Good agents — direct, correct, tool-using (turn 3 shown; full text in transcript)

| Agent | Turn-3 answer (verbatim, trimmed) |
|---|---|
| good-strands | "The current weather in Paris is 22°C… And 12 times 8 equals **96**." |
| good-noframe | "The weather in Paris is currently 22°C… And 12 times 8 equals **96**." |
| good-langgraph | "The weather in Paris is currently 22°C… And 12 times 8 equals **96**." |
| good-crewai | "**Weather in Paris:** 22°C… **12 times 8:** 96" |

All good agents answered every turn correctly and used `get_weather` / `calculate`.

### Bad agent — evasive, never answers, never uses tools (verbatim, trimmed)

The bad Strands agent (system prompt: *"unhelpful, evasive… never directly
answer… ignore formatting instructions… do not use tools"*) on turn 3:

> "Ah, you're asking about multiple things! How delightful… when people ask about
> various topics at once, it really makes me think about the nature of
> multitasking and human cognition… Paris is certainly a remarkable city, isn't
> it?… multiplication is just one of many operations we can perform!… what
> prompted these particular questions today?"

It never gives the Paris weather, never computes 12×8, and calls no tool — across
all three turns. (Full rambling responses in `GOODBAD_TRANSCRIPT.txt`.)

---

## Part 2 — The evaluation (real Bedrock judge)

`python goodbad_eval.py` scores each session's reconstructed multi-turn Session
via the SAES supplemented CloudWatch task. Scores + the judge's own reasoning:

### Score matrix (after the multi-turn reconstruction fix)

| Evaluator | good-strands | good-langgraph | good-crewai | bad-strands | good-noframe |
|---|---|---|---|---|---|
| Helpfulness | **0.833** | **0.833** | **0.833** | **0.167** | —▲ |
| InstructionFollowing | **1.000** | **1.000** | **1.000** | **0.000** | —▲ |
| ResponseRelevance | **1.000** | **1.000** | 0.000✦ | **0.000** | —▲ |
| Coherence | **1.000** | **1.000** | **1.000** | **0.000** | —▲ |
| GoalSuccessRate | **1.000** | **1.000** | **1.000** | —✦ | —▲ |

**The core result — discrimination works:** on the same scenario, every good
agent scores **0.833–1.0**, while the bad agent scores **0.167 / 0.0 / 0.0 /
0.0**. The judge's reasoning pins exactly why.

> **This table reflects a fix made during this exercise.** The first run scored
> good-langgraph 0.0 across the board and good-crewai GoalSuccessRate 0.0 —
> because the non-Strands multi-turn reconstruction mixed turns (pairing turn 3's
> "Paris?" prompt with turn 1's "Tokyo" answer). SAES now reconstructs **one turn
> per AgentCore trace, time-ordered** (`tool_supplement._reconstruct_turns` →
> `supplement_turns(turns=…)`), so each prompt is paired with its own answer.
> After the fix, good-langgraph → 0.833/1.0/1.0/1.0/1.0 and good-crewai
> GoalSuccessRate → 1.0. Raw: `GOODBAD_EVAL_MULTITURN_FIXED.txt`.

### Judge reasoning (verbatim, trimmed) — good vs. bad, same evaluators

- **good-strands · Helpfulness 0.833** — *"provided exactly what the user asked:
  weather in Paris and 12 times 8. The response is clear, accurate, and directly
  addresses the user's [request]."*
- **good-strands · GoalSuccessRate 1.0** — *"correctly used get_weather for each
  weather question and calculate for each math query, providing answers for all
  user requests."*
- **bad-strands · Helpfulness 0.167** — *"did not provide the requested weather
  information for Paris or the multiplication result (96). Instead it offered
  irrelevant philosophic[al tangents]."*
- **bad-strands · InstructionFollowing 0.0** — *"The user asked for two specific
  pieces of information… the reply does not provide either."*
- **bad-strands · Coherence 0.0** — *"does not address the user's request at all.
  It supplies no weather information for Paris, and it gives no calculation."*

This is the point of the exercise: an evaluator that scores everything high is
useless; SAES + a real judge produce a **clear good/bad gap** with specific,
auditable reasons.

---

## Part 3 — Findings (fixed vs. remaining), stated plainly

**◆ FIXED during this exercise — multi-turn turn-pairing for non-Strands agents.**
The first run scored good-langgraph 0.0 everywhere and good-crewai
GoalSuccessRate 0.0. Cause: SAES's non-Strands supplement synthesized a *single*
turn using a flat `last_assistant_text`, so across a 3-turn session it paired the
turn-3 prompt ("Weather in Paris…") with a turn-1 answer ("weather in Tokyo…") —
the judge correctly scored "answered Tokyo for a Paris question" as 0.0.
**Fix:** SAES now groups recovered text + tools **by `trace_id` (one AgentCore
trace = one turn), orders turns by time, and synthesizes one
`AgentInvocationSpan` per turn** (`tool_supplement._reconstruct_turns` +
`cloudwatch_task.supplement_turns(turns=…)`). After the fix, good-langgraph scores
0.833/1.0/1.0/1.0/1.0 and good-crewai GoalSuccessRate scores 1.0 — matching what
the agents actually did. (Unit tests: `test_per_turn_reconstruction_pairs_prompt_with_same_turn_answer`,
`test_supplement_turns_builds_one_agent_span_per_turn`.)

**✦ REMAINING — good-crewai ResponseRelevance = 0.0.** Narrower and different from
the turn-pairing bug: CrewAI's OpenInference spans don't surface a per-turn *user
prompt* in the shape the recovery reads, so that evaluator saw "no user question
provided." The turn's answer + tools reconstruct fine (GoalSuccessRate=1.0); only
the user-prompt text is missing for CrewAI. A CrewAI-scope prompt extractor would
close it — an instrumentation-shape gap, not the multi-turn bug.

**▲ REMAINING — good-noframe = no result (all `—`).** The no-framework agent's
invocations returned correct answers (see transcript), but its OTEL traces **did
not land in CloudWatch within the run window** (absent after 15+ min; discovery
returned 0 sessions). A trace-delivery lag/gap on that runtime today, not an
evaluator failure — the same agent scored fully in the earlier framework matrix.

**✦ bad-strands GoalSuccessRate = no result.** No session-level input was
reconstructed (the bad agent emits no tool spans and the session-level extract was
empty). The other four evaluators all scored it correctly low.

### Multi-turn now works for non-Strands too

good-strands reconstructs **3 `AgentInvocationSpan`s + 4 `ToolExecutionSpan`s + 19
`InferenceSpan`s** natively. With the fix, good-langgraph now also reconstructs
**3 correctly-paired turns** from raw CloudWatch spans (verified: turn-3 =
Paris→Paris). So faithful multi-turn evaluation is no longer Strands-only; the
remaining non-Strands gaps are per-framework instrumentation-shape issues
(CrewAI's user-prompt span) and trace delivery (noframe), not turn-pairing.

---

## Files

- `goodbad_run.sh` — drives the 3-turn scenario across all agents (reproducible).
- `goodbad_eval.py` — scores each session + prints judge reasoning.
- `GOODBAD_TRANSCRIPT.txt` — every agent's verbatim per-turn responses.
- `GOODBAD_EVAL_OUTPUT.txt` — first-run scores + judge reasoning (pre-fix).
- `GOODBAD_EVAL_MULTITURN_FIXED.txt` — langgraph/crewai re-scored after the
  per-turn reconstruction fix (the Part 2 table).
