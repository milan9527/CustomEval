"""Ground-truth dataset loading (SPEC §5).

JSONL, one record per session/trace, keyed by sessionId. Each evaluator reads
only its field (expectedResponse / assertions / expectedTrajectory); fields it
cannot use are reported as ignored by the native evaluators.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config.schema import GroundTruthRef


@dataclass
class GroundTruthRecord:
    expected_response: str | None = None
    assertions: list[str] | None = None
    expected_trajectory: list[str] | None = None


_EMPTY = GroundTruthRecord()


@dataclass
class GroundTruthSet:
    by_session: dict[str, GroundTruthRecord] = field(default_factory=dict)

    def for_session(self, session_id: str) -> GroundTruthRecord:
        return self.by_session.get(session_id, _EMPTY)

    def __bool__(self) -> bool:
        return bool(self.by_session)


def load_ground_truth(ref: GroundTruthRef | None) -> GroundTruthSet:
    if ref is None:
        return GroundTruthSet()
    path = Path(ref.path)
    if not path.is_file():
        raise FileNotFoundError(f"ground truth dataset not found: {path}")

    by_session: dict[str, GroundTruthRecord] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        sid = record.get("sessionId") or record.get("session_id")
        if not sid:
            continue
        by_session[str(sid)] = GroundTruthRecord(
            expected_response=record.get("expectedResponse"),
            assertions=record.get("assertions"),
            expected_trajectory=record.get("expectedTrajectory"),
        )
    return GroundTruthSet(by_session=by_session)


__all__ = ["GroundTruthRecord", "GroundTruthSet", "load_ground_truth"]
