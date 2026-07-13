"""T14 — session-completion tracker (span-quiescence). Injected clock, no waits."""

import json

from saes.online import SessionTracker, TrackerState

MIN = 60_000  # ms per minute


def test_in_progress_session_not_ready():
    t = SessionTracker(timeout_minutes=30)
    now = 100 * MIN
    t.observe([("s1", now - 5 * MIN)])  # last span 5 min ago, timeout 30
    assert t.ready_sessions(now) == []


def test_quiescent_session_is_ready():
    t = SessionTracker(timeout_minutes=30)
    now = 100 * MIN
    t.observe([("s1", now - 31 * MIN)])  # quiet for 31 min > 30
    assert t.ready_sessions(now) == ["s1"]


def test_boundary_exactly_at_timeout_is_ready():
    t = SessionTracker(timeout_minutes=30)
    now = 100 * MIN
    t.observe([("s1", now - 30 * MIN)])  # exactly timeout -> ready (>=)
    assert t.ready_sessions(now) == ["s1"]


def test_scored_session_never_reappears():
    t = SessionTracker(timeout_minutes=10)
    now = 100 * MIN
    t.observe([("s1", now - 20 * MIN)])
    assert t.ready_sessions(now) == ["s1"]
    t.mark_scored(["s1"])
    assert t.ready_sessions(now) == []
    # even if a later cycle still sees it quiescent
    assert t.ready_sessions(now + 100 * MIN) == []


def test_ready_but_not_marked_can_retry():
    """ready_sessions does not auto-mark — a failed score retries next cycle."""
    t = SessionTracker(timeout_minutes=10)
    now = 100 * MIN
    t.observe([("s1", now - 20 * MIN)])
    assert t.ready_sessions(now) == ["s1"]
    # not marked -> still ready
    assert t.ready_sessions(now) == ["s1"]


def test_observe_takes_latest_timestamp():
    t = SessionTracker(timeout_minutes=10)
    now = 100 * MIN
    t.observe([("s1", now - 20 * MIN)])  # would be ready
    t.observe([("s1", now - 1 * MIN)])   # newer span arrived -> back in progress
    assert t.ready_sessions(now) == []


def test_multiple_sessions_mixed():
    t = SessionTracker(timeout_minutes=15)
    now = 1000 * MIN
    t.observe([
        ("done1", now - 20 * MIN),
        ("active", now - 2 * MIN),
        ("done2", now - 16 * MIN),
    ])
    assert set(t.ready_sessions(now)) == {"done1", "done2"}


def test_state_persists_across_restart(tmp_path):
    path = tmp_path / "state.json"
    t1 = SessionTracker(timeout_minutes=10, state_path=path)
    now = 100 * MIN
    t1.observe([("s1", now - 20 * MIN)])
    t1.mark_scored(["s1"])
    assert path.is_file()

    # new tracker from the same state file -> s1 still scored
    t2 = SessionTracker(timeout_minutes=10, state_path=path)
    assert t2.is_scored("s1")
    t2.observe([("s1", now - 20 * MIN)])
    assert t2.ready_sessions(now) == []


def test_scored_count():
    t = SessionTracker(timeout_minutes=10)
    t.observe([("a", 0), ("b", 0)])
    t.mark_scored(["a", "b"])
    assert t.scored_count == 2


def test_state_roundtrip():
    st = TrackerState(last_seen={"s1": 123}, scored={"s0"})
    st2 = TrackerState.from_dict(json.loads(json.dumps(st.to_dict())))
    assert st2.last_seen == {"s1": 123}
    assert st2.scored == {"s0"}
