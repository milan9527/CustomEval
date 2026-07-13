"""Session-completion tracker (SPEC §8.2, T14).

Sessions have no explicit end marker in OTEL traces, so the online worker
treats a session as complete only when no new span has arrived for
`timeout_minutes` after its latest span (span-quiescence timeout, matching
managed AgentCore's SessionConfig). This module decides, given the current
(session_id, last_span_ms) observations and a clock, which sessions are
complete-and-unscored and therefore ready to evaluate — and never returns a
session twice.

State (last-seen per session, scored set) is kept here and optionally persisted
to a JSON file so a worker restart neither re-scores nor loses progress.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrackerState:
    """Persistable tracker state."""

    # session_id -> last observed span epoch ms
    last_seen: dict[str, int] = field(default_factory=dict)
    # session_ids already scored (never re-scored)
    scored: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {"last_seen": self.last_seen, "scored": sorted(self.scored)}

    @classmethod
    def from_dict(cls, data: dict) -> TrackerState:
        return cls(
            last_seen=dict(data.get("last_seen", {})),
            scored=set(data.get("scored", [])),
        )


class SessionTracker:
    """Tracks span quiescence and scored state to select completed sessions."""

    def __init__(
        self,
        timeout_minutes: float,
        state_path: str | Path | None = None,
    ):
        self._timeout_ms = int(timeout_minutes * 60_000)
        self._state_path = Path(state_path) if state_path else None
        self._state = self._load_state()

    # --- observation & selection --------------------------------------------

    def observe(self, sessions: list[tuple[str, int]]) -> None:
        """Record the latest-span timestamps from a discovery cycle."""
        for sid, last_ms in sessions:
            prev = self._state.last_seen.get(sid)
            if prev is None or last_ms > prev:
                self._state.last_seen[sid] = last_ms

    def ready_sessions(self, now_ms: int) -> list[str]:
        """Return session ids that are quiescent (complete) and not yet scored.

        A session is ready when `now_ms - last_seen >= timeout` and it hasn't
        been scored. Does NOT mark them scored — call `mark_scored` after a
        successful evaluation so a mid-cycle failure can be retried next cycle.
        """
        ready: list[str] = []
        for sid, last_ms in self._state.last_seen.items():
            if sid in self._state.scored:
                continue
            if now_ms - last_ms >= self._timeout_ms:
                ready.append(sid)
        return ready

    def mark_scored(self, session_ids: list[str]) -> None:
        """Mark sessions scored so they never re-enter the pipeline."""
        for sid in session_ids:
            self._state.scored.add(sid)
        self._persist()

    # --- introspection -------------------------------------------------------

    def is_scored(self, session_id: str) -> bool:
        return session_id in self._state.scored

    @property
    def scored_count(self) -> int:
        return len(self._state.scored)

    # --- persistence ---------------------------------------------------------

    def _load_state(self) -> TrackerState:
        if self._state_path and self._state_path.is_file():
            return TrackerState.from_dict(json.loads(self._state_path.read_text()))
        return TrackerState()

    def _persist(self) -> None:
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self._state.to_dict(), indent=2))


__all__ = ["SessionTracker", "TrackerState"]
