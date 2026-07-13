"""SAES ingestion (SPEC §7) — thin factory over native strands_evals mappers.

Framework-agnostic: any agent emitting OTEL GenAI-convention spans is
evaluable. SAES delegates normalization to native session mappers and only
owns a local-file reader for offline/CI runs.
"""

from .source import load_sessions, load_sessions_from_file

__all__ = ["load_sessions", "load_sessions_from_file"]
