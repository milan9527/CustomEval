"""SAES configuration schema (SPEC §12).

Pydantic models mirroring the YAML config surface. Secrets are never stored
literally: `JudgeModelConfig.api_key_env` names an environment variable, and
`resolved_api_key()` reads it on demand. Serialization redacts the key.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JudgeProvider(str, Enum):
    OPENAI_COMPATIBLE = "openai_compatible"
    BEDROCK = "bedrock"
    STRANDS = "strands"


class StructuredOutputMode(str, Enum):
    JSON_SCHEMA = "json_schema"
    TOOL_CALL = "tool_call"
    PROMPT = "prompt"


class SamplingConfig(BaseModel):
    """Online-mode sampling controls (SPEC §8.2)."""

    model_config = ConfigDict(extra="forbid")

    percentage: float = Field(100.0, ge=0.0, le=100.0)
    max_per_minute: int | None = Field(None, ge=1)
    filters: list[str] = Field(default_factory=list)


class SessionConfig(BaseModel):
    """Session-completion detection for online evaluation (SPEC §8.2).

    A session is treated as complete when no new span has arrived for
    `timeout_minutes` after its latest span (span-quiescence timeout), matching
    managed AgentCore's SessionConfig. Set near the agent's typical session
    duration.
    """

    model_config = ConfigDict(extra="forbid")

    timeout_minutes: float = Field(30.0, gt=0)


class JudgeModelConfig(BaseModel):
    """LLM-as-a-Judge model selection (SPEC §3).

    The differentiator: any OpenAI-compatible endpoint can serve as the judge.
    """

    model_config = ConfigDict(extra="forbid")

    provider: JudgeProvider = JudgeProvider.OPENAI_COMPATIBLE
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = Field(60.0, gt=0)
    max_retries: int = Field(3, ge=0)
    structured_output: StructuredOutputMode = StructuredOutputMode.JSON_SCHEMA

    @model_validator(mode="after")
    def _check_openai_compatible_requirements(self) -> JudgeModelConfig:
        if self.provider is JudgeProvider.OPENAI_COMPATIBLE and not self.base_url:
            raise ValueError(
                "judge.base_url is required when provider is 'openai_compatible'"
            )
        return self

    def resolved_api_key(self) -> str | None:
        """Read the API key from the named env var. Never stored on the model."""
        import os

        if self.api_key_env is None:
            return None
        key = os.environ.get(self.api_key_env)
        if key is None:
            raise RuntimeError(
                f"judge.api_key_env='{self.api_key_env}' is set but that "
                "environment variable is not defined"
            )
        return key


class CloudWatchSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_group_names: list[str] = Field(default_factory=list)
    service_names: list[str] = Field(default_factory=list)
    agent_name: str | None = None
    region: str | None = None
    lookback_days: int = Field(30, ge=1)
    filter: str | None = None

    @model_validator(mode="after")
    def _need_a_target(self) -> CloudWatchSource:
        if not self.log_group_names and not self.agent_name:
            raise ValueError(
                "cloudwatch data source needs either log_group_names or agent_name"
            )
        return self


class DataSourceConfig(BaseModel):
    """Where agent traces come from (SPEC §7). Framework-agnostic: the contract
    is OTEL GenAI spans, not a specific SDK."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["cloudwatch", "otlp_file", "langfuse", "live"]
    cloudwatch: CloudWatchSource | None = None
    path: str | None = None
    mapper: str | None = Field(
        None,
        description="Override the session mapper; defaults to auto-detection.",
    )

    @model_validator(mode="after")
    def _check_type_fields(self) -> DataSourceConfig:
        if self.type == "cloudwatch" and self.cloudwatch is None:
            raise ValueError("data source type 'cloudwatch' requires a `cloudwatch` block")
        if self.type == "otlp_file" and not self.path:
            raise ValueError("data source type 'otlp_file' requires a `path`")
        return self


class EvaluatorRef(BaseModel):
    """A built-in evaluator id (e.g. 'Builtin.Helpfulness') or a custom evaluator.

    In M1 only the built-in string form is exercised; the object form reserves
    the shape for custom LLM/code evaluators (SPEC §6)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["builtin", "llm", "code"] = "builtin"
    level: Literal["session", "trace", "tool"] | None = None
    judge_override: JudgeModelConfig | None = None
    instructions: str | None = None
    scale: str | None = None
    base_template: str | None = None


class GroundTruthRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class CloudWatchSink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_group: str
    metrics_namespace: str = "SAES/Evaluations"
    dimensions: list[str] = Field(default_factory=list)


class LocalSink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    json_path: str | None = None
    html_report: str | None = None


class ResultsSinkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cloudwatch: CloudWatchSink | None = None
    local: LocalSink | None = None


class EvaluationConfig(BaseModel):
    """Top-level SAES config (SPEC §12)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    description: str | None = None
    mode: Literal["on_demand", "online"] = "on_demand"
    data_source: DataSourceConfig = Field(alias="dataSource")
    evaluators: list[EvaluatorRef]
    judge: JudgeModelConfig
    sampling: SamplingConfig | None = None
    session: SessionConfig | None = None
    ground_truth: GroundTruthRef | None = Field(None, alias="groundTruth")
    results_sink: ResultsSinkConfig | None = Field(None, alias="resultsSink")
    gate: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _mode_consistency(self) -> EvaluationConfig:
        if self.mode == "online":
            # online mode needs sampling + session-completion defaults
            if self.sampling is None:
                self.sampling = SamplingConfig()
            if self.session is None:
                self.session = SessionConfig()
        return self
