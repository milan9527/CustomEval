"""T1 config layer tests (SPEC §12)."""

import pytest

from saes.config import (
    ConfigError,
    EvaluationConfig,
    JudgeProvider,
    parse_config,
    redacted_dict,
    validate_semantics,
)
from saes.config.loader import load_config

SPEC_EXAMPLE = {
    "name": "my-agent-quality",
    "mode": "on_demand",
    "dataSource": {
        "type": "cloudwatch",
        "cloudwatch": {
            "log_group_names": ["/aws/bedrock-agentcore/my-agent"],
            "service_names": ["my-agent.DEFAULT"],
            "region": "us-east-1",
        },
    },
    "judge": {
        "provider": "openai_compatible",
        "model": "gpt-4.1",
        "base_url": "https://llm-gateway.internal/v1",
        "api_key_env": "SAES_JUDGE_API_KEY",
        "params": {"temperature": 0.0, "max_tokens": 1024},
        "structured_output": "json_schema",
    },
    "evaluators": [
        "Builtin.Helpfulness",
        "Builtin.Correctness",
        "Builtin.ToolSelectionAccuracy",
        "Builtin.GoalSuccessRate",
        {"id": "hipaa_compliance", "type": "llm", "level": "trace", "scale": "binary"},
    ],
    "groundTruth": {"path": "./ground_truth.jsonl"},
    "resultsSink": {
        "cloudwatch": {
            "log_group": "/aws/saes/evaluations",
            "metrics_namespace": "SAES/Evaluations",
            "dimensions": ["agentId", "evaluatorId"],
        },
        "local": {"html_report": "./out/report.html"},
    },
    "gate": ["Builtin.Helpfulness.avg >= 0.8", "Builtin.Correctness.avg >= 0.9"],
}


def test_parses_spec_example():
    cfg = parse_config(SPEC_EXAMPLE)
    assert cfg.name == "my-agent-quality"
    assert cfg.judge.provider is JudgeProvider.OPENAI_COMPATIBLE
    assert cfg.data_source.type == "cloudwatch"
    assert len(cfg.evaluators) == 5
    # compact string form normalized to builtin refs
    assert cfg.evaluators[0].id == "Builtin.Helpfulness"
    assert cfg.evaluators[0].type == "builtin"
    # object form preserved
    assert cfg.evaluators[-1].id == "hipaa_compliance"
    assert cfg.evaluators[-1].type == "llm"
    assert cfg.ground_truth.path == "./ground_truth.jsonl"


def test_roundtrip_by_alias():
    cfg = parse_config(SPEC_EXAMPLE)
    dumped = cfg.model_dump(mode="json", by_alias=True)
    reparsed = parse_config(dumped)
    assert reparsed.name == cfg.name
    assert reparsed.judge.model == cfg.judge.model


def test_unknown_builtin_rejected():
    bad = dict(SPEC_EXAMPLE)
    bad["evaluators"] = ["Builtin.NotAReal_Evaluator"]
    with pytest.raises(ConfigError, match="unknown built-in evaluator"):
        parse_config(bad)


def test_openai_compatible_requires_base_url():
    bad = dict(SPEC_EXAMPLE)
    bad["judge"] = {"provider": "openai_compatible", "model": "gpt-4.1"}
    with pytest.raises(Exception, match="base_url is required"):
        parse_config(bad)


def test_bedrock_judge_needs_no_base_url():
    ok = dict(SPEC_EXAMPLE)
    ok["judge"] = {"provider": "bedrock", "model": "us.anthropic.claude-sonnet-5"}
    cfg = parse_config(ok)
    assert cfg.judge.provider is JudgeProvider.BEDROCK


def test_cloudwatch_source_needs_target():
    bad = dict(SPEC_EXAMPLE)
    bad["dataSource"] = {"type": "cloudwatch", "cloudwatch": {"region": "us-east-1"}}
    with pytest.raises(Exception, match="log_group_names or agent_name"):
        parse_config(bad)


def test_secret_never_serialized():
    """api_key_env holds the var NAME, not the key; the key is never on the model."""
    cfg = parse_config(SPEC_EXAMPLE)
    dumped_str = str(cfg.model_dump(mode="json"))
    # even if the env var were set, the literal key must not appear in the model
    assert "SAES_JUDGE_API_KEY" in dumped_str  # the var name is fine
    # the JudgeModelConfig has no field that could hold a literal key
    assert not hasattr(cfg.judge, "api_key")


def test_resolved_api_key_reads_env(monkeypatch):
    cfg = parse_config(SPEC_EXAMPLE)
    monkeypatch.setenv("SAES_JUDGE_API_KEY", "sk-secret-123")
    assert cfg.judge.resolved_api_key() == "sk-secret-123"


def test_resolved_api_key_missing_env_raises(monkeypatch):
    cfg = parse_config(SPEC_EXAMPLE)
    monkeypatch.delenv("SAES_JUDGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not defined"):
        cfg.judge.resolved_api_key()


def test_redacted_dict_strips_url_userinfo():
    data = dict(SPEC_EXAMPLE)
    data["judge"] = dict(SPEC_EXAMPLE["judge"])
    data["judge"]["base_url"] = "https://user:pass@llm.internal/v1"
    cfg = parse_config(data)
    red = redacted_dict(cfg)
    assert "pass" not in red["judge"]["base_url"]
    assert red["judge"]["base_url"] == "https://llm.internal/v1"


def test_ground_truth_evaluator_without_dataset_warns():
    data = dict(SPEC_EXAMPLE)
    data = {**SPEC_EXAMPLE, "evaluators": ["Builtin.Correctness"]}
    data.pop("groundTruth", None)
    cfg = parse_config(data)
    warnings = validate_semantics(cfg)
    assert any("reference-free" in w for w in warnings)


def test_online_mode_defaults_sampling_and_session():
    data = {**SPEC_EXAMPLE, "mode": "online"}
    cfg = parse_config(data)
    # online mode fills in sampling + session-completion defaults (SPEC §8.2)
    assert cfg.sampling is not None
    assert cfg.session is not None
    assert cfg.session.timeout_minutes == 30.0


def test_session_timeout_override():
    data = {**SPEC_EXAMPLE, "mode": "online", "session": {"timeout_minutes": 5}}
    cfg = parse_config(data)
    assert cfg.session.timeout_minutes == 5.0


def test_on_demand_has_no_session_default():
    cfg = parse_config(SPEC_EXAMPLE)  # mode defaults to on_demand
    assert cfg.session is None


def test_load_config_from_file(tmp_path):
    import yaml

    p = tmp_path / "eval.yaml"
    p.write_text(yaml.safe_dump(SPEC_EXAMPLE))
    cfg = load_config(p)
    assert isinstance(cfg, EvaluationConfig)
    assert cfg.name == "my-agent-quality"


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent/eval.yaml")
