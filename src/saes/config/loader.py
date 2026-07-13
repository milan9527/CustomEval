"""Config loading, validation, and safe serialization (SPEC §12, T1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import EvaluationConfig, EvaluatorRef


def _builtin_ids() -> frozenset[str]:
    """Valid built-in evaluator ids — derived from the evaluator registry (the
    single source of truth) so the config validator never drifts from what
    `resolve_evaluator` actually supports (incl. trajectory scorers). Imported
    lazily to avoid pulling the evaluator stack when only parsing config."""
    from ..evaluators.registry import available_builtins

    return frozenset(available_builtins())

# Built-in evaluators that consume ground-truth fields (SPEC §5).
GROUND_TRUTH_EVALUATORS: frozenset[str] = frozenset(
    {"Builtin.Correctness", "Builtin.GoalSuccessRate"}
)


class ConfigError(ValueError):
    """Raised when a config is structurally valid but semantically wrong."""


def _normalize_evaluators(raw: list[Any]) -> list[dict[str, Any]]:
    """Allow the compact string form (`- Builtin.Helpfulness`) alongside the
    object form, normalizing both to EvaluatorRef dicts."""
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"id": item, "type": "builtin"})
        elif isinstance(item, dict):
            out.append(item)
        else:
            raise ConfigError(f"evaluator entry must be a string or mapping, got {type(item)}")
    return out


def load_config(path: str | Path) -> EvaluationConfig:
    """Load and validate a SAES config from YAML."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    return parse_config(data)


def parse_config(data: dict[str, Any]) -> EvaluationConfig:
    """Validate an already-parsed config mapping."""
    data = dict(data)
    if "evaluators" in data and isinstance(data["evaluators"], list):
        data["evaluators"] = _normalize_evaluators(data["evaluators"])
    config = EvaluationConfig.model_validate(data)
    _semantic_checks(config)
    return config


def _semantic_checks(config: EvaluationConfig) -> None:
    warnings = validate_semantics(config)
    # Hard errors: unknown built-in ids (validated against the registry).
    valid_ids = _builtin_ids()
    for ev in config.evaluators:
        if ev.type == "builtin" and ev.id not in valid_ids:
            raise ConfigError(
                f"unknown built-in evaluator '{ev.id}'. "
                f"Valid ids: {', '.join(sorted(valid_ids))}"
            )
    # Soft warnings are surfaced via return of validate_semantics; the loader
    # itself does not raise on them.
    _ = warnings


def validate_semantics(config: EvaluationConfig) -> list[str]:
    """Return non-fatal warnings (e.g. ground-truth evaluator without a dataset)."""
    warnings: list[str] = []
    has_gt = config.ground_truth is not None
    for ev in config.evaluators:
        if (
            ev.type == "builtin"
            and ev.id in GROUND_TRUTH_EVALUATORS
            and not has_gt
        ):
            warnings.append(
                f"evaluator '{ev.id}' can use ground truth but no `groundTruth` "
                "source is configured; it will run in reference-free mode."
            )
    if config.mode == "online" and config.gate:
        warnings.append(
            "`gate` rules are ignored in online mode (gates apply to on_demand runs)."
        )
    return warnings


def redacted_dict(config: EvaluationConfig) -> dict[str, Any]:
    """Serialize a config for logging/provenance with secrets removed.

    The API key is never stored on the model (only the env var *name* is), but
    we defensively redact base_url too since it can carry embedded credentials.
    """
    data = config.model_dump(mode="json", by_alias=True)
    judge = data.get("judge")
    if isinstance(judge, dict):
        if judge.get("base_url"):
            judge["base_url"] = _redact_url(judge["base_url"])
        # api_key_env is a var name, not a secret — keep it for provenance.
    for ev in data.get("evaluators", []):
        jo = ev.get("judge_override") if isinstance(ev, dict) else None
        if isinstance(jo, dict) and jo.get("base_url"):
            jo["base_url"] = _redact_url(jo["base_url"])
    return data


def _redact_url(url: str) -> str:
    """Strip any userinfo (user:pass@) from a URL, keep scheme+host+path."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    if parts.username or parts.password:
        netloc = parts.hostname or ""
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        parts = parts._replace(netloc=netloc)
    return urlunsplit(parts)


__all__ = [
    "GROUND_TRUTH_EVALUATORS",
    "ConfigError",
    "load_config",
    "parse_config",
    "validate_semantics",
    "redacted_dict",
    "EvaluatorRef",
]
