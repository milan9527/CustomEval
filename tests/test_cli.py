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
    monkeypatch.setattr("saes.online.scoring.make_scorer", lambda cfg: (lambda ids: list(ids)))

    cfg = _write_online_config(tmp_path)
    result = runner.invoke(app, ["serve", "-c", str(cfg), "--once"])
    assert result.exit_code == 0, result.output
    assert "cycle: ready=1 scored=1" in result.output


def test_serve_rejects_non_cloudwatch(tmp_path):
    cfg = _write_config(tmp_path)  # otlp_file source
    result = runner.invoke(app, ["serve", "-c", str(cfg), "--once"])
    assert result.exit_code == 2
    assert "requires dataSource.type: cloudwatch" in result.output
