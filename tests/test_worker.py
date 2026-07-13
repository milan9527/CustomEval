"""T15 — online worker cycle orchestration. Injected discover/score/clock."""

from saes.config.schema import (
    DataSourceConfig,
    EvaluationConfig,
    JudgeModelConfig,
    SamplingConfig,
    SessionConfig,
)
from saes.online import OnlineWorker, RateLimiter
from saes.online.worker import _sampled

MIN = 60_000


def _config(percentage=100.0, max_per_minute=None, timeout=10.0):
    return EvaluationConfig(
        name="online",
        mode="online",
        dataSource=DataSourceConfig(
            type="cloudwatch",
            cloudwatch={"log_group_names": ["/aws/x"], "region": "us-east-1"},
        ),
        judge=JudgeModelConfig(provider="bedrock", model="m"),
        evaluators=[{"id": "Builtin.Helpfulness", "type": "builtin"}],
        sampling=SamplingConfig(percentage=percentage, max_per_minute=max_per_minute),
        session=SessionConfig(timeout_minutes=timeout),
    )


def _worker(cfg, observations, score_fn, logs=None):
    return OnlineWorker(
        cfg,
        discover=lambda: observations,
        score=score_fn,
        log=(logs.append if logs is not None else None),
    )


# ---- basic cycle ------------------------------------------------------------

def test_cycle_scores_completed_sessions():
    now = 100 * MIN
    obs = [("done", now - 20 * MIN), ("active", now - 1 * MIN)]
    scored_calls = []

    def score(ids):
        scored_calls.append(list(ids))
        return list(ids)  # all succeed

    w = _worker(_config(), obs, score)
    r = w.run_cycle(now)
    assert r.ready == ["done"]  # active is still in progress
    assert r.scored == ["done"]
    assert scored_calls == [["done"]]


def test_completed_session_scored_once_across_cycles():
    now = 100 * MIN
    obs = [("done", now - 20 * MIN)]
    calls = []
    w = _worker(_config(), obs, lambda ids: (calls.append(list(ids)) or list(ids)))
    w.run_cycle(now)
    w.run_cycle(now + 5 * MIN)  # same session still present, already scored
    assert calls == [["done"]]  # scored exactly once


# ---- sampling ---------------------------------------------------------------

def test_sampling_percentage_zero_scores_nothing():
    now = 100 * MIN
    obs = [("a", now - 20 * MIN), ("b", now - 20 * MIN)]
    w = _worker(_config(percentage=0.0), obs, lambda ids: list(ids))
    r = w.run_cycle(now)
    assert r.sampled == []
    assert r.scored == []


def test_sampling_is_deterministic():
    ids = [f"session-{i}" for i in range(200)]
    a = _sampled(ids, 50.0)
    b = _sampled(ids, 50.0)
    assert a == b  # stable across calls
    assert 0 < len(a) < len(ids)  # roughly half, not all/none


# ---- rate cap ---------------------------------------------------------------

def test_rate_cap_defers_excess():
    now = 100 * MIN
    obs = [(f"s{i}", now - 20 * MIN) for i in range(5)]
    logs = []
    w = _worker(_config(max_per_minute=2), obs, lambda ids: list(ids), logs=logs)
    r = w.run_cycle(now)
    assert len(r.scored) == 2
    assert len(r.deferred) == 3
    assert any("deferred" in m for m in logs)  # deferral logged, not silent


def test_rate_limiter_rolling_window():
    lim = RateLimiter(max_per_minute=3)
    assert lim.take(0, 5) == 3          # 3 allowed at t=0
    assert lim.take(30_000, 5) == 0     # window full within 60s
    assert lim.take(61_000, 5) == 3     # first 3 aged out after 60s


def test_rate_cap_none_is_unlimited():
    now = 100 * MIN
    obs = [(f"s{i}", now - 20 * MIN) for i in range(50)]
    w = _worker(_config(max_per_minute=None), obs, lambda ids: list(ids))
    r = w.run_cycle(now)
    assert len(r.scored) == 50


# ---- failure handling -------------------------------------------------------

def test_failed_score_retries_next_cycle():
    now = 100 * MIN
    obs = [("s1", now - 20 * MIN)]
    attempts = {"n": 0}

    def score(ids):
        attempts["n"] += 1
        return [] if attempts["n"] == 1 else list(ids)  # fail first, succeed second

    w = _worker(_config(), obs, score)
    r1 = w.run_cycle(now)
    assert r1.scored == [] and r1.errored == ["s1"]  # not marked -> retryable
    r2 = w.run_cycle(now + MIN)
    assert r2.scored == ["s1"]  # retried and succeeded
