"""Structured-output extraction and repair for judge responses (SPEC §3.3).

The AgentCore-published templates instruct the judge to return a pure JSON
object (sometimes fenced in triple backticks) with a `reason`/`reasoning`
field and a `score`/`verdict` field. On endpoints that support native
schema/tool enforcement we get clean JSON; on prompt-only endpoints we must
extract it robustly. This module is enforcement-strategy agnostic: it parses
whatever text came back and, if that fails, signals the caller to re-ask.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Field aliases seen across the AgentCore templates.
_REASON_KEYS = ("reason", "reasoning", "explanation")
_SCORE_KEYS = ("score", "verdict", "label")

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class ParseError(ValueError):
    """The judge response could not be parsed into a verdict."""


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from judge output.

    Handles: bare JSON, ```json fenced blocks, and JSON embedded in prose
    (first balanced `{...}` span).
    """
    if not text or not text.strip():
        raise ParseError("empty judge response")

    candidates: list[str] = []
    # 1. fenced blocks first (templates ask for triple-backtick JSON)
    candidates.extend(m.group(1) for m in _FENCE_RE.finditer(text))
    # 2. the whole string
    candidates.append(text.strip())
    # 3. first balanced object span
    span = _first_balanced_object(text)
    if span is not None:
        candidates.append(span)

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    raise ParseError("no JSON object found in judge response")


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_verdict_fields(obj: dict[str, Any]) -> tuple[str, Any]:
    """Pull (reason, raw_score) out of a parsed verdict object using the
    known key aliases. Raises ParseError if a score-like field is absent."""
    reason = ""
    for k in _REASON_KEYS:
        if k in obj and obj[k] is not None:
            reason = str(obj[k])
            break

    for k in _SCORE_KEYS:
        if k in obj and obj[k] is not None:
            return reason, obj[k]

    raise ParseError(
        f"verdict missing a score field (looked for {_SCORE_KEYS}); got keys {list(obj)}"
    )


# The JSON schema SAES advertises to schema-capable endpoints. Individual
# templates may override the `score` enum; this is the generic shape.
GENERIC_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "string"},
    },
    "required": ["reasoning", "score"],
    "additionalProperties": False,
}


__all__ = [
    "GENERIC_VERDICT_SCHEMA",
    "ParseError",
    "extract_json",
    "parse_verdict_fields",
]
