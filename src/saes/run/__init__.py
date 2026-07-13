"""SAES on-demand run layer (T5) — native Experiment orchestration + CI gate."""

from .gate import GateCheck, GateError, GateReport, evaluate_gate
from .ground_truth import GroundTruthRecord, GroundTruthSet, load_ground_truth
from .runner import RunResult, run_on_demand

__all__ = [
    "GateCheck",
    "GateError",
    "GateReport",
    "GroundTruthRecord",
    "GroundTruthSet",
    "RunResult",
    "evaluate_gate",
    "load_ground_truth",
    "run_on_demand",
]
