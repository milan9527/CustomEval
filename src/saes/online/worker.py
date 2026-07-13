"""Online evaluation worker (SPEC §8.2, T15).

Composes the pieces M1/M2 already built — it orchestrates, it does not
reimplement scoring, discovery, or emission:

  discover (session id + last-seen) → SessionTracker selects completed+unscored
  → sampling (percentage) + rolling-window rate cap → score via the native
  Experiment (per-session, using the CloudWatch provider's read+map) → emit to
  CloudWatch (EMF + JSON) → mark scored.

The core loop is split into `run_cycle()` (pure orchestration, injectable clock
and hooks) so it is testable without real sleeping or AWS. `serve()` wraps it
in a polling loop.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from ..config.schema import EvaluationConfig
from ..online.session_tracker import SessionTracker


@dataclass
class CycleResult:
    """Outcome of one worker cycle."""

    ready: list[str] = field(default_factory=list)
    sampled: list[str] = field(default_factory=list)
    scored: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)  # eligible but capped this cycle
    errored: list[str] = field(default_factory=list)


def _sampled(session_ids: list[str], percentage: float) -> list[str]:
    """Deterministic percentage sampling by session-id hash (stable across
    cycles so the same session isn't flapped in/out of the sample)."""
    if percentage >= 100.0:
        return list(session_ids)
    if percentage <= 0.0:
        return []
    threshold = percentage / 100.0
    out: list[str] = []
    for sid in session_ids:
        h = hashlib.sha256(sid.encode()).digest()
        # first 8 bytes -> fraction in [0,1)
        frac = int.from_bytes(h[:8], "big") / 2**64
        if frac < threshold:
            out.append(sid)
    return out


class RateLimiter:
    """Rolling-window cap: at most `max_per_minute` scores per 60s of wall time
    (by the injected clock), enforced across cycles."""

    def __init__(self, max_per_minute: int | None):
        self._max = max_per_minute
        self._window: deque[int] = deque()  # epoch ms of recent scores

    def take(self, now_ms: int, want: int) -> int:
        """How many of `want` may proceed now, updating the window."""
        if self._max is None:
            return want
        cutoff = now_ms - 60_000
        while self._window and self._window[0] < cutoff:
            self._window.popleft()
        allowed = max(0, self._max - len(self._window))
        grant = min(want, allowed)
        for _ in range(grant):
            self._window.append(now_ms)
        return grant


class OnlineWorker:
    """Drives online evaluation cycles."""

    def __init__(
        self,
        config: EvaluationConfig,
        *,
        discover: Callable[[], Iterable[tuple[str, int]]],
        score: Callable[[list[str]], list[str]],
        tracker: SessionTracker | None = None,
        log: Callable[[str], None] | None = None,
    ):
        """
        discover: returns [(session_id, last_span_ms)] for the current window.
        score:    scores the given session ids, returns the ids successfully
                  scored (the caller wires this to the runner + CloudWatch sink).
        """
        self.config = config
        self._discover = discover
        self._score = score
        timeout = config.session.timeout_minutes if config.session else 30.0
        self._tracker = tracker or SessionTracker(timeout_minutes=timeout)
        sampling = config.sampling
        self._percentage = sampling.percentage if sampling else 100.0
        self._limiter = RateLimiter(sampling.max_per_minute if sampling else None)
        self._log = log or (lambda _m: None)

    def run_cycle(self, now_ms: int) -> CycleResult:
        observations = list(self._discover())
        self._tracker.observe(observations)

        ready = self._tracker.ready_sessions(now_ms)
        sampled = _sampled(ready, self._percentage)

        grant = self._limiter.take(now_ms, len(sampled))
        to_score = sampled[:grant]
        deferred = sampled[grant:]
        if deferred:
            self._log(
                f"rate cap: {len(deferred)} eligible session(s) deferred to a "
                f"later cycle (max_per_minute={self.config.sampling.max_per_minute})"
            )

        scored: list[str] = []
        errored: list[str] = []
        if to_score:
            scored = list(self._score(to_score))
            errored = [s for s in to_score if s not in scored]
            if scored:
                self._tracker.mark_scored(scored)  # only successes marked
            self._log(f"scored {len(scored)}/{len(to_score)} session(s) this cycle")

        return CycleResult(
            ready=ready,
            sampled=sampled,
            scored=scored,
            deferred=deferred,
            errored=errored,
        )


__all__ = ["CycleResult", "OnlineWorker", "RateLimiter"]
