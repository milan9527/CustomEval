"""SAES command-line interface (SPEC §8, §11, T7/T15).

    saes eval   RUNTIME_ID [--judge-model ...] [--evaluators ...] [--html out.html]
    saes run    --config eval.yaml [--dataset gt.jsonl] [--json out.json] [--html out.html]
    saes doctor [--data-source dump.jsonl] [--judge eval.yaml]
    saes init   [--agent-type customer-service|rag|tool-heavy] [--out eval.yaml]
    saes serve  RUNTIME_ID [--sampling 100] [--interval 60] [--once]   (or --config online.yaml)

`eval` is the one-liner: give it an AgentCore Runtime id and it evaluates that
runtime's recent CloudWatch traces — no YAML, no ground truth. `run` exits
non-zero when the CI gate fails (SPEC §8.1). `serve` runs the online worker loop
(SPEC §8.2).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
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


# Reference-free built-ins: score with no ground truth, so they are the safe
# default for the zero-config `saes eval`. (Correctness/GoalSuccessRate/trajectory
# matchers need expectedResponse/assertions/expectedTrajectory — use `saes run`.)
_REFERENCE_FREE_EVALUATORS = [
    "Builtin.Helpfulness",
    "Builtin.Coherence",
    "Builtin.Conciseness",
    "Builtin.Faithfulness",
    "Builtin.InstructionFollowing",
    "Builtin.ResponseRelevance",
    "Builtin.ContextRelevance",
    "Builtin.Harmfulness",
    "Builtin.Refusal",
    "Builtin.Stereotyping",
    "Builtin.ToolSelectionAccuracy",
    "Builtin.ToolParameterAccuracy",
]


def _select_evaluator_ids(evaluators: str | None, all_evaluators: bool) -> list[str]:
    """Resolve the evaluator id list for the zero-config commands: explicit
    --evaluators wins, else --all (every built-in except trajectory matchers,
    which need ground truth), else the reference-free default. Validates ids and
    exits(2) on an unknown one."""
    from .evaluators.registry import available_builtins

    if evaluators:
        ev_ids = [e.strip() for e in evaluators.split(",") if e.strip()]
    elif all_evaluators:
        ev_ids = [e for e in available_builtins() if not e.startswith("Builtin.Trajectory")]
    else:
        ev_ids = list(_REFERENCE_FREE_EVALUATORS)
    unknown = [e for e in ev_ids if e not in set(available_builtins())]
    if unknown:
        typer.secho(f"unknown evaluator id(s): {', '.join(unknown)}",
                    fg=typer.colors.RED, err=True)
        typer.echo("run `saes eval --list-evaluators` to see valid ids.", err=True)
        raise typer.Exit(code=2)
    return ev_ids


@contextlib.contextmanager
def _quiet_judge_retry_noise():
    """Silence the per-retry ERROR the Strands tool executor logs when a judge
    model emits a slightly malformed structured-output tool name (e.g. gpt-oss
    returning 'HhelpfulnessRating'). The judge retries and still scores; the line
    is benign noise, not a failure. Restored on exit."""
    lg = logging.getLogger("strands.tools.executors._executor")
    saved = lg.level
    lg.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        lg.setLevel(saved)


@app.command()
def eval(  # noqa: A001 - deliberately named `eval` for the CLI verb
    runtime: str = typer.Argument(
        None,
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
    lookback_days: int = typer.Option(
        7, "--lookback-days", "--days",
        help="How many days of traces to scan (default 7; increase for older sessions)",
    ),
    evaluators: str | None = typer.Option(
        None, "--evaluators", "-e",
        help="Comma-separated evaluator ids. Default: the 12 reference-free built-ins. "
             "Use --all for every built-in, or --list-evaluators to see them.",
    ),
    all_evaluators: bool = typer.Option(
        False, "--all",
        help="Run all 13 built-in evaluators (adds Correctness + GoalSuccessRate; "
             "they score better with ground truth via `saes run`).",
    ),
    sampling: float = typer.Option(
        100.0, "--sampling", "--sample",
        min=0.0, max=100.0,
        help="Percent of discovered sessions to score (default 100). Deterministic "
             "by session id, matching AgentCore Evaluations' sampling.",
    ),
    list_evaluators: bool = typer.Option(
        False, "--list-evaluators", help="List the available evaluator ids and exit."
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
        SamplingConfig,
    )
    from .evaluators.registry import available_builtins
    from .report import build_report, write_html, write_json
    from .run import run_on_demand

    if list_evaluators:
        typer.secho("built-in evaluators:", fg=typer.colors.GREEN)
        for eid in available_builtins():
            tag = "" if eid in _REFERENCE_FREE_EVALUATORS else "  (needs ground truth → saes run)"
            typer.echo(f"  {eid}{tag}")
        typer.echo(
            "\ncustom evaluators: define type: llm / type: code in a config (saes run)."
        )
        raise typer.Exit(code=0)

    if not runtime:
        typer.secho("error: RUNTIME is required (or use --list-evaluators)",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    ev_ids = _select_evaluator_ids(evaluators, all_evaluators)

    log_group = _runtime_log_group(runtime)
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
        sampling=SamplingConfig(percentage=sampling),
    )

    typer.secho(f"evaluating {log_group}", fg=typer.colors.GREEN)
    sample_note = "" if sampling >= 100.0 else f"  |  sampling {sampling:g}%"
    typer.echo(
        f"  judge: {judge_model}  |  last {lookback_days}d{sample_note}  |  "
        f"{len(ev_ids)} evaluator(s): {', '.join(ev_ids)}"
    )

    try:
        with _quiet_judge_retry_noise():
            result = asyncio.run(run_on_demand(cfg))
    except Exception as exc:  # noqa: BLE001 - surface a clean message, not a traceback
        msg = str(exc)
        if "ResourceNotFoundException" in type(exc).__name__ or "does not exist" in msg:
            typer.secho(
                f"\nlog group not found: {log_group}", fg=typer.colors.RED, err=True
            )
            typer.echo(
                "  • check the runtime id (e.g. 'myagent-XyZ123' — no '-DEFAULT' needed)\n"
                "  • or the agent has never emitted traces to CloudWatch yet.",
                err=True,
            )
        else:
            typer.secho(f"\nevaluation failed: {msg}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    doc = build_report(result)

    # No sessions discovered in the window is the common "n=0 everywhere" cause —
    # detect it explicitly (aggregates still carry one n=0 row per evaluator) and
    # give an actionable message instead of a wall of zeros.
    if not result.session_ids:
        typer.secho(
            f"\nno sessions found in {log_group} over the last {lookback_days} day(s).",
            fg=typer.colors.YELLOW,
        )
        typer.echo(
            "  • widen the window:  saes eval "
            f"{runtime} --days 30\n"
            "  • or the agent hasn't run recently / traces aren't indexed yet "
            "(~90s after an invoke)."
        )
        raise typer.Exit(code=0)

    typer.echo(
        f"\n{doc.config_name}  (judge: {doc.judge_model})  "
        f"[{len(result.session_ids)} session(s)]"
    )
    for ev_id, stats in doc.aggregates.items():
        line = (
            f"  {ev_id:32s} avg={stats['avg']:.3f}  "
            f"pass={stats['pass_rate'] * 100:.0f}%  n={int(stats['n'])}"
        )
        if stats.get("errored"):
            line += f"  errored={int(stats['errored'])}"
        typer.echo(line)
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
    runtime: str = typer.Argument(
        None,
        help="AgentCore Runtime id (e.g. 'myagent-XyZ123') — the zero-config path. "
             "Omit and pass --config to use a full YAML instead.",
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Online-mode config YAML (alternative to RUNTIME)"
    ),
    # zero-config options (used when RUNTIME is given; mirror `saes eval`)
    judge_model: str = typer.Option("openai.gpt-oss-20b-1:0", "--judge-model"),
    judge_base_url: str = typer.Option(
        "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1", "--judge-base-url"
    ),
    api_key_env: str = typer.Option("SAES_JUDGE_API_KEY", "--api-key-env"),
    region: str = typer.Option("us-east-1", "--region"),
    evaluators: str | None = typer.Option(
        None, "--evaluators", "-e",
        help="Comma-separated evaluator ids (default: the 12 reference-free built-ins)",
    ),
    all_evaluators: bool = typer.Option(False, "--all", help="Run all 13 built-in evaluators"),
    sampling: float = typer.Option(
        100.0, "--sampling", "--sample", min=0.0, max=100.0,
        help="Percent of completed sessions to score (default 100)",
    ),
    timeout_minutes: float = typer.Option(
        5.0, "--session-timeout",
        help="A session is 'complete' after this many minutes with no new span",
    ),
    results_log_group: str | None = typer.Option(
        None, "--results-log-group",
        help="CloudWatch log group for results (default: /aws/saes/<runtime>-results)",
    ),
    interval: float = typer.Option(60.0, "--interval", help="Seconds between polling cycles"),
    state: Path | None = typer.Option(None, "--state", help="JSON state file (survives restarts)"),
    once: bool = typer.Option(False, "--once", help="Run a single cycle and exit"),
    print_scores: bool = typer.Option(
        False, "--print-scores",
        help="Also print each scored batch's per-evaluator scores in the terminal "
             "(scores always go to CloudWatch either way)",
    ),
) -> None:
    """Continuously evaluate a live agent's new CloudWatch traffic (online mode).

    Zero-config: just give an AgentCore Runtime id — `saes serve myagent-XyZ123`.
    It polls the runtime's log group, detects completed sessions (span-quiescence
    timeout), samples + scores them, and writes results to CloudWatch. Each
    session is scored at most once. For CI gates / custom evaluators / a non-
    AgentCore log group, pass a full YAML with `--config` instead.
    """
    import time as _time

    from .config import load_config
    from .config.schema import (
        CloudWatchSink,
        CloudWatchSource,
        DataSourceConfig,
        EvaluationConfig,
        EvaluatorRef,
        JudgeModelConfig,
        ResultsSinkConfig,
        SamplingConfig,
        SessionConfig,
    )
    from .ingest.cloudwatch import build_provider, discover_sessions_with_last_seen
    from .online import OnlineWorker, SessionTracker
    from .online.scoring import make_scorer

    if config is not None:
        cfg = load_config(config)
        if cfg.data_source.type != "cloudwatch":
            typer.secho(
                "serve requires dataSource.type: cloudwatch", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=2)
    elif runtime:
        # zero-config: build an online config in-memory from just the runtime id
        log_group = _runtime_log_group(runtime)
        ev_ids = _select_evaluator_ids(evaluators, all_evaluators)
        rid = runtime.split("/")[-1].removesuffix("-DEFAULT")
        sink_group = results_log_group or f"/aws/saes/{rid}-results"
        cfg = EvaluationConfig(
            name=f"serve-{rid}",
            mode="online",
            data_source=DataSourceConfig(
                type="cloudwatch",
                cloudwatch=CloudWatchSource(
                    log_group_names=[log_group], region=region, lookback_days=1
                ),
            ),
            evaluators=[EvaluatorRef(id=e) for e in ev_ids],
            judge=JudgeModelConfig(
                model=judge_model, base_url=judge_base_url, api_key_env=api_key_env
            ),
            sampling=SamplingConfig(percentage=sampling),
            session=SessionConfig(timeout_minutes=timeout_minutes),
            results_sink=ResultsSinkConfig(
                cloudwatch=CloudWatchSink(
                    log_group=sink_group, metrics_namespace="SAES/Evaluations",
                    dimensions=["agentId", "evaluatorId"],
                )
            ),
        )
        typer.secho(f"serving {log_group}", fg=typer.colors.GREEN)
        typer.echo(f"  results → {sink_group}  |  {len(ev_ids)} evaluator(s)")
    else:
        typer.secho("error: give a RUNTIME id or --config", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    def _print_report(doc) -> None:
        typer.echo(f"  scores ({doc.judge_model}):")
        for ev_id, stats in doc.aggregates.items():
            typer.echo(
                f"    {ev_id:30s} avg={stats['avg']:.3f}  "
                f"pass={stats['pass_rate'] * 100:.0f}%  n={int(stats['n'])}"
            )

    provider = build_provider(cfg.data_source.cloudwatch)
    timeout = cfg.session.timeout_minutes if cfg.session else 30.0
    tracker = SessionTracker(timeout_minutes=timeout, state_path=state)
    worker = OnlineWorker(
        cfg,
        discover=lambda: discover_sessions_with_last_seen(provider, cfg.data_source.cloudwatch),
        score=make_scorer(cfg, on_report=_print_report if print_scores else None),
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
        with _quiet_judge_retry_noise():
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
