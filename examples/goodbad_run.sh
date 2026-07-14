#!/usr/bin/env bash
# Drive a shared 3-turn tool-calling scenario across all live SAES agents:
#   good: strands / noframe / langgraph / crewai   +   bad: strands (saesbad)
# Each agent's 3 turns share one AgentCore session id (multi-turn), so SAES
# reconstructs a multi-turn Session from CloudWatch. Captures verbatim responses.
set -uo pipefail
source /home/ec2-user/saes_run_venv/bin/activate

STAMP="$(date +%s)"
OUT="/home/ec2-user/saes_run/GOODBAD_TRANSCRIPT.txt"
: > "$OUT"

# agent dir  |  a label used to build a session id
AGENTS=(
  "strands_tools:good-strands"
  "noframework_tools:good-noframe"
  "langgraph_tools:good-langgraph"
  "crewai_tools:good-crewai"
  "bad_agent:bad-strands"
)

PROMPTS=(
  "What is the weather in Tokyo?"
  "What is 15% of 240?"
  "Weather in Paris, and what is 12 times 8?"
)

for entry in "${AGENTS[@]}"; do
  dir="${entry%%:*}"; label="${entry##*:}"
  # AgentCore session ids need to be reasonably long; keep alnum + stamp + label
  sid="$(echo "${label}${STAMP}sessionpadding123456" | tr -d '-')"
  echo "==================================================================" | tee -a "$OUT"
  echo "AGENT: $label   dir=$dir   session=$sid" | tee -a "$OUT"
  echo "==================================================================" | tee -a "$OUT"
  for i in "${!PROMPTS[@]}"; do
    p="${PROMPTS[$i]}"
    echo "" | tee -a "$OUT"
    echo "--- turn $((i+1)): $p" | tee -a "$OUT"
    resp="$(cd "/home/ec2-user/saes_run/agents/$dir" && \
            timeout 150 agentcore invoke --session-id "$sid" "{\"prompt\": \"$p\"}" 2>/dev/null \
            | sed -n '/^Response:/,$p' | tail -n +2)"
    echo "$resp" | tee -a "$OUT"
  done
  echo "" | tee -a "$OUT"
  echo "SESSION_ID_FOR_$label=$sid" | tee -a "$OUT"
  echo "" | tee -a "$OUT"
done
echo "DONE. transcript -> $OUT"
