"""SAES command-line interface (SPEC §8, §11, T7/T15).

    saes eval   RUNTIME_ID [--judge-model ...] [--evaluators ...] [--html out.html]
    saes run    --config eval.yaml [--dataset gt.jsonl] [--json out.json] [--html out.html]
    saes doctor [--data-source dump.jsonl] [--judge eval.yaml]
    saes init   [--agent-type customer-service|rag|tool-heavy] [--out eval.yaml]
    saes serve  --config online.yaml [--interval 60] [--state state.json] [--once]

`eval` is the one-liner: give it an AgentCore Runtime id and it evaluates that
runtime's recent CloudWatch traces — no YAML, no ground truth. `run` exits
non-zero when the CI gate fails (SPEC §8.1). `serve` runs the online worker loop
(SPEC §8.2).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Strands Agent Evaluation Suite")


# --- run ---------------------------------------------------------------------


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="SAES config YAML"),
    dataset: Path | None = typer.Option(None, "--dataset", help="Ground-truth JSONL override"),
    json_out: Path | None = typer.Option(None, "--json", help="Write JSON report here"),
    html_out: Path | None = typer.Option(None, "--html", help="Write HTML report here"),
) -> None:
    """Run an on-demand evaluation and (optionally) gate on thresholds."""
    from .config import load_config
    from .config.loader import validate_semantics
    from .config.schema import GroundTruthRef
    from .report import build_report, write_html, write_json
    from .run import evaluate_gate, run_on_demand
    from .run.gate import GateError

    cfg = load_config(config)
    if dataset is not None:
        cfg.ground_truth = GroundTruthRef(path=str(dataset))

    for warning in validate_semantics(cfg):
        typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW, err=True)

    result = asyncio.run(run_on_demand(cfg))

    gate_report = None
    if cfg.gate:
        try:
            gate_report = evaluate_gate(cfg.gate, result.aggregates)
        except GateError as exc:
            typer.secho(f"gate error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc

    doc = build_report(result, gate=gate_report)

    # console summary
    typer.echo(f"\n{doc.config_name}  (judge: {doc.judge_model})")
    for ev_id, stats in doc.aggregates.items():
        line = (
            f"  {ev_id:32s} avg={stats['avg']:.3f}  "
            f"pass={stats['pass_rate'] * 100:.0f}%  n={int(stats['n'])}"
        )
        if stats["errored"]:
            line += f"  errored={int(stats['errored'])}"
        typer.echo(line)

    if json_out is not None:
        write_json(doc, json_out)
        typer.echo(f"\nJSON  → {json_out}")
    if html_out is not None:
        write_html(doc, html_out)
        typer.echo(f"HTML  → {html_out}")

    sink = cfg.results_sink.cloudwatch if cfg.results_sink else None
    if sink is not None:
        from .report import emit_to_cloudwatch

        region = cfg.data_source.cloudwatch.region if cfg.data_source.cloudwatch else None
        try:
            emit_to_cloudwatch(doc, sink, region=region)
            typer.echo(f"CloudWatch → {sink.log_group} (ns: {sink.metrics_namespace})")
        except Exception as exc:  # noqa: BLE001 - don't fail the run on emit error
            typer.secho(f"warning: CloudWatch emit failed: {exc}", fg=typer.colors.YELLOW, err=True)

    if gate_report is not None:
        _print_gate(gate_report)
        if not gate_report.passed:
            raise typer.Exit(code=1)


def _print_gate(gate_report) -> None:
    status = "PASSED" if gate_report.passed else "FAILED"
    color = typer.colors.GREEN if gate_report.passed else typer.colors.RED
    typer.secho(f"\nGATE {status}", fg=color, bold=True)
    for c in gate_report.checks:
        mark = "✓" if c.passed else "✗"
        typer.secho(
            f"  {mark} {c.rule}  (actual={c.actual:.3f})",
            fg=typer.colors.GREEN if c.passed else typer.colors.RED,
        )


# --- doctor ------------------------------------------------------------------


@app.command()
def doctor(
    data_source: Path | None = typer.Option(None, "--data-source", help="OTLP/JSONL span dump"),
    judge: Path | None = typer.Option(None, "--judge", help="Config YAML: probe its judge"),
) -> None:
    """Preflight checks (SPEC §7.1a, §3.5).

    --data-source: per-field OTEL coverage + whether sessions reconstruct.
    --judge:       verify the configured judge supports tool-calling/structured
                   output (required by the native evaluators).
    """
    if judge is not None:
        _probe_judge_config(judge)
    if data_source is None and judge is None:
        typer.secho(
            "nothing to check: pass --data-source and/or --judge",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=2)
    if data_source is None:
        return

    from .ingest import load_sessions_from_file
    from .ingest.conformance import check_conformance

    try:
        report = check_conformance(data_source)
    except Exception as exc:  # noqa: BLE001 - report cleanly, don't traceback
        typer.secho(f"failed to read spans: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"spans read: {report.n_spans}")
    typer.echo("field coverage:")
    for fc in report.fields:
        mark = "✓" if fc.covered else "✗"
        color = typer.colors.GREEN if fc.covered else typer.colors.YELLOW
        typer.secho(
            f"  {mark} {fc.label:32s} {fc.present}/{fc.total}  ({', '.join(fc.aliases)})",
            fg=color,
        )

    # then try actual reconstruction via the native mappers
    try:
        sessions = load_sessions_from_file(data_source)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"mapping failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if not sessions:
        typer.secho(
            "\nno sessions reconstructed — see missing (✗) fields above.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    typer.secho(
        f"\nOK — {len(sessions)} session(s) reconstructed", fg=typer.colors.GREEN
    )
    for s in sessions:
        sid = getattr(s, "session_id", "?")
        n_traces = len(getattr(s, "traces", []) or [])
        typer.echo(f"  {sid}: {n_traces} trace(s)")


def _probe_judge_config(config_path: Path) -> None:
    """Probe the judge in a config for structured-output support (SPEC §3.5)."""
    from .config import load_config
    from .judge import probe_judge

    cfg = load_config(config_path)
    typer.echo(f"probing judge: {cfg.judge.provider.value} / {cfg.judge.model}")
    result = probe_judge(cfg.judge)
    if result.supported:
        typer.secho(f"  ✓ {result.detail}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"  ✗ {result.detail}", fg=typer.colors.RED, err=True)
        typer.secho(
            "  the judge must support tool calling / structured output "
            "(SPEC §3.5); a text-only endpoint cannot drive the native evaluators.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)


# --- eval (one-liner: just an AgentCore Runtime id) --------------------------


def _runtime_log_group(runtime: str) -> str:
    """Map an AgentCore Runtime id to its CloudWatch log group.

    Accepts a bare id ('myagent-XXXX'), an id with the DEFAULT endpoint suffix,
    or a full '/aws/bedrock-agentcore/runtimes/...' path (passed through)."""
    runtime = runtime.strip()
    if runtime.startswith("/aws/"):
        return runtime
    if not runtime.endswith("-DEFAULT"):
        runtime = f"{runtime}-DEFAULT"
    return f"/aws/bedrock-agentcore/runtimes/{runtime}"


@app.command()
def eval(  # noqa: A001 - deliberately named `eval` for the CLI verb
    runtime: str = typer.Argument(
        ...,
        help="AgentCore Runtime id (e.g. 'myagent-XyZ123'), or its full CloudWatch log group",
    ),
    judge_model: str = typer.Option(
        "openai.gpt-oss-20b-1:0", "--judge-model",
        help="Judge model id (default: a Bedrock OpenAI-compatible model)",
    ),
    judge_base_url: str = typer.Option(
        "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1", "--judge-base-url",
        help="Judge endpoint base URL",
    ),
    api_key_env: str = typer.Option(
        "SAES_JUDGE_API_KEY", "--api-key-env", help="Env var holding the judge API key"
    ),
    region: str = typer.Option("us-east-1", "--region"),
    lookback_days: int = typer.Option(1, "--lookback-days", help="How far back to look for traces"),
    evaluators: str = typer.Option(
        "Builtin.Helpfulness,Builtin.Coherence,Builtin.Conciseness,Builtin.ResponseRelevance",
        "--evaluators",
        help="Comma-separated evaluator ids (default: reference-free, no ground truth needed)",
    ),
    html_out: Path | None = typer.Option(None, "--html", help="Write HTML report here"),
    json_out: Path | None = typer.Option(None, "--json", help="Write JSON report here"),
) -> None:
    """Evaluate an AgentCore Runtime's recent CloudWatch traces — just give the id.

    Builds the config in-memory (no YAML, no ground truth, no manual session
    discovery): derives the runtime's log group, discovers its sessions, and
    scores them with the chosen judge. For ground-truth evaluators (Correctness,
    Trajectory*Match, …) or CI gates, use a full config with `saes run`.
    """
    from .config.schema import (
        CloudWatchSource,
        DataSourceConfig,
        EvaluationConfig,
        EvaluatorRef,
        JudgeModelConfig,
    )
    from .report import build_report, write_html, write_json
    from .run import run_on_demand

    log_group = _runtime_log_group(runtime)
    ev_ids = [e.strip() for e in evaluators.split(",") if e.strip()]
    cfg = EvaluationConfig(
        name=f"eval-{runtime.split('/')[-1]}",
        mode="on_demand",
        data_source=DataSourceConfig(
            type="cloudwatch",
            cloudwatch=CloudWatchSource(
                log_group_names=[log_group], region=region, lookback_days=lookback_days
            ),
        ),
        evaluators=[EvaluatorRef(id=e) for e in ev_ids],
        judge=JudgeModelConfig(
            model=judge_model, base_url=judge_base_url, api_key_env=api_key_env
        ),
    )

    typer.secho(f"evaluating {log_group}", fg=typer.colors.GREEN)
    typer.echo(f"  judge: {judge_model}  |  evaluators: {', '.join(ev_ids)}")

    result = asyncio.run(run_on_demand(cfg))
    doc = build_report(result)

    typer.echo(f"\n{doc.config_name}  (judge: {doc.judge_model})")
    if not doc.aggregates:
        typer.secho(
            "  no sessions scored — no traces found in the lookback window, or the "
            "session hasn't finished. Try a larger --lookback-days.",
            fg=typer.colors.YELLOW,
        )
    for ev_id, stats in doc.aggregates.items():
        typer.echo(
            f"  {ev_id:32s} avg={stats['avg']:.3f}  "
            f"pass={stats['pass_rate'] * 100:.0f}%  n={int(stats['n'])}"
        )
    if json_out is not None:
        write_json(doc, json_out)
        typer.echo(f"\nJSON  → {json_out}")
    if html_out is not None:
        write_html(doc, html_out)
        typer.echo(f"HTML  → {html_out}")


# --- init --------------------------------------------------------------------

_RECOMMENDED = {
    "customer-service": [
        "Builtin.Helpfulness",
        "Builtin.GoalSuccessRate",
        "Builtin.InstructionFollowing",
    ],
    "rag": [
        "Builtin.Correctness",
        "Builtin.Faithfulness",
        "Builtin.ResponseRelevance",
    ],
    "tool-heavy": [
        "Builtin.ToolSelectionAccuracy",
        "Builtin.ToolParameterAccuracy",
        "Builtin.Helpfulness",
    ],
}


@app.command()
def init(
    agent_type: str = typer.Option(
        "customer-service", "--agent-type",
        help="customer-service | rag | tool-heavy",
    ),
    out: Path = typer.Option(Path("eval.yaml"), "--out", "-o"),
) -> None:
    """Scaffold a starter config with evaluators recommended for the agent type."""
    import yaml

    evaluators = _RECOMMENDED.get(agent_type)
    if evaluators is None:
        typer.secho(
            f"unknown agent type '{agent_type}'; choose from {sorted(_RECOMMENDED)}",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=2)

    scaffold = {
        "name": f"{agent_type}-quality",
        "mode": "on_demand",
        "dataSource": {"type": "otlp_file", "path": "./traces.otlp.jsonl"},
        "judge": {
            "provider": "openai_compatible",
            "model": "gpt-4.1",
            "base_url": "https://your-endpoint/v1",
            "api_key_env": "SAES_JUDGE_API_KEY",
            "params": {"temperature": 0.0},
        },
        "evaluators": evaluators,
        "resultsSink": {"local": {"html_report": "./out/report.html"}},
    }
    if out.exists():
        typer.secho(f"refusing to overwrite existing {out}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    out.write_text(yaml.safe_dump(scaffold, sort_keys=False))
    typer.secho(f"wrote {out}  ({', '.join(evaluators)})", fg=typer.colors.GREEN)
    typer.echo(
        "next: set your judge endpoint + SAES_JUDGE_API_KEY, then verify it with\n"
        f"  saes doctor --judge {out}\n"
        "(the judge must support tool calling / structured output — SPEC §3.5)"
    )


# --- serve (online worker) ---------------------------------------------------


@app.command()
def serve(
    config: Path = typer.Option(..., "--config", "-c", help="Online-mode config YAML"),
    interval: float = typer.Option(60.0, "--interval", help="Seconds between polling cycles"),
    state: Path | None = typer.Option(None, "--state", help="JSON state file (survives restarts)"),
    once: bool = typer.Option(False, "--once", help="Run a single cycle and exit"),
) -> None:
    """Run the online evaluation worker loop (SPEC §8.2).

    Polls the CloudWatch data source, detects completed sessions (span-quiescence
    timeout), samples + scores them, and emits results to CloudWatch. Each
    session is scored at most once.
    """
    import time as _time

    from .config import load_config
    from .ingest.cloudwatch import build_provider, discover_sessions_with_last_seen
    from .online import OnlineWorker, SessionTracker
    from .online.scoring import make_scorer

    cfg = load_config(config)
    if cfg.data_source.type != "cloudwatch":
        typer.secho(
            "serve requires dataSource.type: cloudwatch", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2)

    provider = build_provider(cfg.data_source.cloudwatch)
    timeout = cfg.session.timeout_minutes if cfg.session else 30.0
    tracker = SessionTracker(timeout_minutes=timeout, state_path=state)
    worker = OnlineWorker(
        cfg,
        discover=lambda: discover_sessions_with_last_seen(provider, cfg.data_source.cloudwatch),
        score=make_scorer(cfg),
        tracker=tracker,
        log=lambda m: typer.echo(f"  {m}"),
    )

    typer.secho(
        f"serving online eval for '{cfg.name}' "
        f"(timeout={timeout}m, sampling={cfg.sampling.percentage}%)",
        fg=typer.colors.GREEN,
    )
    while True:
        now_ms = int(_time.time() * 1000)
        result = worker.run_cycle(now_ms)
        typer.echo(
            f"cycle: ready={len(result.ready)} scored={len(result.scored)} "
            f"deferred={len(result.deferred)} errored={len(result.errored)}"
        )
        if once:
            break
        _time.sleep(interval)


def main() -> None:  # pragma: no cover - entrypoint
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
