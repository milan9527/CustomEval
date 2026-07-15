"""T7 CLI tests via typer's CliRunner."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from saes.cli import app
from saes.run.runner import RunResult

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


# ---- init -------------------------------------------------------------------

def test_init_scaffolds_config(tmp_path):
    out = tmp_path / "eval.yaml"
    result = runner.invoke(app, ["init", "--agent-type", "rag", "--out", str(out)])
    assert result.exit_code == 0
    cfg = yaml.safe_load(out.read_text())
    assert cfg["name"] == "rag-quality"
    assert "Builtin.Correctness" in cfg["evaluators"]
    assert cfg["judge"]["provider"] == "openai_compatible"


def test_init_refuses_overwrite(tmp_path):
    out = tmp_path / "eval.yaml"
    out.write_text("existing")
    result = runner.invoke(app, ["init", "--out", str(out)])
    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output


def test_init_unknown_agent_type(tmp_path):
    result = runner.invoke(
        app, ["init", "--agent-type", "nope", "--out", str(tmp_path / "e.yaml")]
    )
    assert result.exit_code == 2
    assert "unknown agent type" in result.output


# ---- doctor -----------------------------------------------------------------

def test_doctor_reports_sessions():
    result = runner.invoke(
        app, ["doctor", "--data-source", str(FIXTURES / "openinference_session.jsonl")]
    )
    assert result.exit_code == 0
    assert "session(s) reconstructed" in result.output
    # T12: per-field coverage is reported
    assert "field coverage:" in result.output
    assert "session id" in result.output


def test_doctor_judge_probe_pass(tmp_path, monkeypatch):
    from saes.judge.probe import ProbeResult

    monkeypatch.setattr(
        "saes.judge.probe_judge",
        lambda cfg: ProbeResult(supported=True, detail="ok", model=cfg.model),
    )
    cfg = _write_config(tmp_path)
    result = runner.invoke(app, ["doctor", "--judge", str(cfg)])
    assert result.exit_code == 0
    assert "probing judge" in result.output


def test_doctor_judge_probe_fail_exits_one(tmp_path, monkeypatch):
    from saes.judge.probe import ProbeResult

    monkeypatch.setattr(
        "saes.judge.probe_judge",
        lambda cfg: ProbeResult(supported=False, detail="text-only", model=cfg.model),
    )
    cfg = _write_config(tmp_path)
    result = runner.invoke(app, ["doctor", "--judge", str(cfg)])
    assert result.exit_code == 1
    assert "text-only" in result.output


def test_doctor_nothing_to_check():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 2
    assert "nothing to check" in result.output


def test_doctor_empty_file_warns(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    result = runner.invoke(app, ["doctor", "--data-source", str(empty)])
    assert result.exit_code == 1
    assert "no sessions" in result.output


# ---- run (runner patched, no real judge) ------------------------------------

def _fake_run_result(avg):
    class _Report:
        detailed_results = [
            [type("O", (), {"score": avg, "reason": "r", "label": "L", "test_pass": avg >= 0.5})()]
        ]
        cases = [{"evaluator": "Builtin.Helpfulness", "name": "s1"}]
        overall_score = avg

    return RunResult(
        config_name="demo",
        judge_model="m",
        report=_Report(),
        evaluator_ids=["Builtin.Helpfulness"],
        session_ids=["s1"],
        aggregates={
            "Builtin.Helpfulness": {"avg": avg, "pass_rate": 1.0, "n": 1.0, "errored": 0.0}
        },
    )


def _write_config(tmp_path, gate=None):
    cfg = {
        "name": "demo",
        "dataSource": {"type": "otlp_file", "path": "x.jsonl"},
        "judge": {"provider": "openai_compatible", "model": "m", "base_url": "https://x/v1"},
        "evaluators": ["Builtin.Helpfulness"],
    }
    if gate:
        cfg["gate"] = gate
    p = tmp_path / "eval.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_run_writes_reports(tmp_path, monkeypatch):
    monkeypatch.setattr("saes.run.run_on_demand", lambda cfg: _async(_fake_run_result(0.9)))
    cfg = _write_config(tmp_path)
    json_out = tmp_path / "r.json"
    html_out = tmp_path / "r.html"
    result = runner.invoke(
        app, ["run", "-c", str(cfg), "--json", str(json_out), "--html", str(html_out)]
    )
    assert result.exit_code == 0, result.output
    assert json_out.exists() and html_out.exists()
    assert "Builtin.Helpfulness" in result.output


def test_run_gate_pass_exit_zero(tmp_path, monkeypatch):
    monkeypatch.setattr("saes.run.run_on_demand", lambda cfg: _async(_fake_run_result(0.9)))
    cfg = _write_config(tmp_path, gate=["Builtin.Helpfulness.avg >= 0.8"])
    result = runner.invoke(app, ["run", "-c", str(cfg)])
    assert result.exit_code == 0
    assert "GATE PASSED" in result.output


def test_run_gate_fail_exit_one(tmp_path, monkeypatch):
    monkeypatch.setattr("saes.run.run_on_demand", lambda cfg: _async(_fake_run_result(0.4)))
    cfg = _write_config(tmp_path, gate=["Builtin.Helpfulness.avg >= 0.8"])
    result = runner.invoke(app, ["run", "-c", str(cfg)])
    assert result.exit_code == 1
    assert "GATE FAILED" in result.output


async def _async(value):
    return value


# ---- eval (one-liner: just a runtime id) ------------------------------------

def test_runtime_log_group_mapping():
    from saes.cli import _runtime_log_group

    assert _runtime_log_group("myagent-ABC") == \
        "/aws/bedrock-agentcore/runtimes/myagent-ABC-DEFAULT"
    # already has the endpoint suffix -> not doubled
    assert _runtime_log_group("myagent-ABC-DEFAULT") == \
        "/aws/bedrock-agentcore/runtimes/myagent-ABC-DEFAULT"
    # full log-group path -> passed through unchanged
    full = "/aws/bedrock-agentcore/runtimes/x-DEFAULT"
    assert _runtime_log_group(full) == full


def test_eval_builds_config_from_runtime_id(tmp_path, monkeypatch):
    """`saes eval <runtime>` needs no YAML/ground truth: it derives the log group,
    defaults to reference-free evaluators, and runs run_on_demand."""
    captured = {}

    def _fake_run(cfg):
        captured["cfg"] = cfg
        return _async(_fake_run_result(0.9))

    monkeypatch.setattr("saes.run.run_on_demand", _fake_run)
    result = runner.invoke(app, ["eval", "myagent-XyZ", "--lookback-days", "2"])
    assert result.exit_code == 0, result.output
    cfg = captured["cfg"]
    # the runtime id became the CloudWatch log group, no ground truth required
    assert cfg.data_source.type == "cloudwatch"
    assert cfg.data_source.cloudwatch.log_group_names == [
        "/aws/bedrock-agentcore/runtimes/myagent-XyZ-DEFAULT"
    ]
    assert cfg.data_source.cloudwatch.lookback_days == 2
    assert cfg.ground_truth is None
    # default = the 12 reference-free built-ins (no ground truth needed)
    assert len(cfg.evaluators) == 12
    assert {e.id for e in cfg.evaluators} == set(_REFERENCE_FREE)
    assert "Builtin.Helpfulness" in result.output


_REFERENCE_FREE = {
    "Builtin.Helpfulness", "Builtin.Coherence", "Builtin.Conciseness",
    "Builtin.Faithfulness", "Builtin.InstructionFollowing", "Builtin.ResponseRelevance",
    "Builtin.ContextRelevance", "Builtin.Harmfulness", "Builtin.Refusal",
    "Builtin.Stereotyping", "Builtin.ToolSelectionAccuracy", "Builtin.ToolParameterAccuracy",
}


def test_eval_all_flag_selects_all_builtins(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "saes.run.run_on_demand",
        lambda cfg: (captured.__setitem__("cfg", cfg), _async(_fake_run_result(0.9)))[1],
    )
    result = runner.invoke(app, ["eval", "myagent-XyZ", "--all"])
    assert result.exit_code == 0, result.output
    ids = {e.id for e in captured["cfg"].evaluators}
    # --all adds the ground-truth built-ins too, but excludes trajectory matchers
    assert "Builtin.Correctness" in ids and "Builtin.GoalSuccessRate" in ids
    assert not any(i.startswith("Builtin.Trajectory") for i in ids)


def test_eval_explicit_evaluators(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "saes.run.run_on_demand",
        lambda cfg: (captured.__setitem__("cfg", cfg), _async(_fake_run_result(0.9)))[1],
    )
    result = runner.invoke(
        app, ["eval", "myagent-XyZ", "-e", "Builtin.Helpfulness,Builtin.Harmfulness"]
    )
    assert result.exit_code == 0, result.output
    assert [e.id for e in captured["cfg"].evaluators] == [
        "Builtin.Helpfulness", "Builtin.Harmfulness"
    ]


def test_eval_sampling_flag_reaches_config(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "saes.run.run_on_demand",
        lambda cfg: (captured.__setitem__("cfg", cfg), _async(_fake_run_result(0.9)))[1],
    )
    result = runner.invoke(app, ["eval", "myagent-XyZ", "--sampling", "25"])
    assert result.exit_code == 0, result.output
    assert captured["cfg"].sampling.percentage == 25.0
    assert "sampling 25%" in result.output


def test_eval_unknown_evaluator_rejected(monkeypatch):
    monkeypatch.setattr("saes.run.run_on_demand", lambda cfg: _async(_fake_run_result(0.9)))
    result = runner.invoke(app, ["eval", "myagent-XyZ", "-e", "Builtin.Nope"])
    assert result.exit_code == 2
    assert "unknown evaluator" in result.output


def test_eval_list_evaluators():
    result = runner.invoke(app, ["eval", "--list-evaluators"])
    assert result.exit_code == 0, result.output
    assert "Builtin.Helpfulness" in result.output
    assert "Builtin.Correctness" in result.output
    # ground-truth ones are tagged
    assert "needs ground truth" in result.output


def test_eval_reports_when_no_sessions(tmp_path, monkeypatch):
    """No sessions discovered -> an actionable message suggesting a wider window,
    not a wall of n=0 rows. (The runner still returns one n=0 aggregate row per
    evaluator, so detection keys off session_ids, not empty aggregates.)"""
    monkeypatch.setattr(
        "saes.run.run_on_demand",
        lambda cfg: _async(_empty_run_result()),
    )
    result = runner.invoke(app, ["eval", "myagent-XyZ"])
    assert result.exit_code == 0, result.output
    assert "no sessions found" in result.output
    assert "--days 30" in result.output          # suggests widening the window
    assert "avg=0.000" not in result.output       # no zero-row wall


def test_eval_lookback_flag_reaches_config(monkeypatch):
    """--days / --lookback-days sets the CloudWatch lookback window."""
    captured = {}
    monkeypatch.setattr(
        "saes.run.run_on_demand",
        lambda cfg: (captured.__setitem__("cfg", cfg), _async(_fake_run_result(0.9)))[1],
    )
    result = runner.invoke(app, ["eval", "myagent-XyZ", "--days", "30"])
    assert result.exit_code == 0, result.output
    assert captured["cfg"].data_source.cloudwatch.lookback_days == 30
    assert "last 30d" in result.output


def test_eval_missing_log_group_clean_error(monkeypatch):
    """A bad runtime id / nonexistent log group -> a clean message + exit 1, not
    a raw ResourceNotFoundException traceback."""
    def _boom(cfg):
        raise RuntimeError(
            "ResourceNotFoundException: Log group '/aws/...' does not exist"
        )
    monkeypatch.setattr("saes.run.run_on_demand", _boom)
    result = runner.invoke(app, ["eval", "typo-agent"])
    assert result.exit_code == 1
    assert "log group not found" in result.output
    assert "check the runtime id" in result.output


def _empty_run_result():
    class _Report:
        # aggregates present with n=0 (what the runner really returns for 0
        # sessions) — the point being detection must NOT rely on empty aggregates
        detailed_results = []
        cases = []
        overall_score = 0.0

    return RunResult(
        config_name="eval-x", judge_model="m", report=_Report(),
        evaluator_ids=["Builtin.Helpfulness"], session_ids=[],
        aggregates={
            "Builtin.Helpfulness": {"avg": 0.0, "pass_rate": 0.0, "n": 0.0, "errored": 0.0}
        },
    )


# ---- serve (online worker, --once, no AWS) ----------------------------------

def _write_online_config(tmp_path):
    cfg = {
        "name": "online",
        "mode": "online",
        "dataSource": {
            "type": "cloudwatch",
            "cloudwatch": {"log_group_names": ["/aws/x"], "region": "us-east-1"},
        },
        "judge": {"provider": "bedrock", "model": "m"},
        "evaluators": ["Builtin.Helpfulness"],
        "session": {"timeout_minutes": 10},
    }
    p = tmp_path / "online.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_serve_once_runs_one_cycle(tmp_path, monkeypatch):
    # stub the CloudWatch provider, discovery, and scorer — no AWS
    monkeypatch.setattr("saes.ingest.cloudwatch.build_provider", lambda cw: object())
    # one completed session (last span 20 min ago, timeout 10)
    import time as _t

    now_ms = int(_t.time() * 1000)
    monkeypatch.setattr(
        "saes.ingest.cloudwatch.discover_sessions_with_last_seen",
        lambda provider, cw: [("s1", now_ms - 20 * 60_000)],
    )
    monkeypatch.setattr(
        "saes.online.scoring.make_scorer",
        lambda cfg, on_report=None: (lambda ids: list(ids)),
    )

    cfg = _write_online_config(tmp_path)
    result = runner.invoke(app, ["serve", "-c", str(cfg), "--once"])
    assert result.exit_code == 0, result.output
    assert "cycle: ready=1 scored=1" in result.output


def test_serve_rejects_non_cloudwatch(tmp_path):
    cfg = _write_config(tmp_path)  # otlp_file source
    result = runner.invoke(app, ["serve", "-c", str(cfg), "--once"])
    assert result.exit_code == 2
    assert "requires dataSource.type: cloudwatch" in result.output


def test_serve_zero_config_from_runtime_id(monkeypatch):
    """`saes serve <runtime>` needs no YAML: it builds the online config from the
    runtime id (log group, default evaluators, an auto-derived results sink)."""
    import time as _t

    captured = {}

    def _scorer(cfg, on_report=None):
        captured["cfg"] = cfg
        return lambda ids: list(ids)

    monkeypatch.setattr("saes.ingest.cloudwatch.build_provider", lambda cw: object())
    now_ms = int(_t.time() * 1000)
    monkeypatch.setattr(
        "saes.ingest.cloudwatch.discover_sessions_with_last_seen",
        lambda provider, cw: [("s1", now_ms - 20 * 60_000)],
    )
    monkeypatch.setattr("saes.online.scoring.make_scorer", _scorer)

    result = runner.invoke(
        app, ["serve", "myagent-XyZ", "--session-timeout", "1", "--once"]
    )
    assert result.exit_code == 0, result.output
    assert "cycle: ready=1 scored=1" in result.output
    cfg = captured["cfg"]
    assert cfg.mode == "online"
    assert cfg.data_source.cloudwatch.log_group_names == [
        "/aws/bedrock-agentcore/runtimes/myagent-XyZ-DEFAULT"
    ]
    # results sink auto-derived from the runtime id, no YAML needed
    assert cfg.results_sink.cloudwatch.log_group == "/aws/saes/myagent-XyZ-results"
    assert cfg.session.timeout_minutes == 1.0
    assert len(cfg.evaluators) == 12  # reference-free default


def test_serve_requires_runtime_or_config():
    result = runner.invoke(app, ["serve", "--once"])
    assert result.exit_code == 2
    assert "give a RUNTIME id or --config" in result.output
