"""T5 ground-truth dataset loading (SPEC §5)."""

import pytest

from saes.config.schema import GroundTruthRef
from saes.run import load_ground_truth


def test_none_ref_yields_empty_set():
    gt = load_ground_truth(None)
    assert not gt
    rec = gt.for_session("anything")
    assert rec.expected_response is None
    assert rec.assertions is None
    assert rec.expected_trajectory is None


def test_loads_jsonl(tmp_path):
    p = tmp_path / "gt.jsonl"
    p.write_text(
        '{"sessionId": "s1", "expectedResponse": "Paris", '
        '"assertions": ["a", "b"], "expectedTrajectory": ["t1", "t2"]}\n'
        '{"sessionId": "s2", "expectedResponse": "London"}\n'
        "\n"  # blank line ignored
    )
    gt = load_ground_truth(GroundTruthRef(path=str(p)))
    assert bool(gt)
    s1 = gt.for_session("s1")
    assert s1.expected_response == "Paris"
    assert s1.assertions == ["a", "b"]
    assert s1.expected_trajectory == ["t1", "t2"]
    s2 = gt.for_session("s2")
    assert s2.expected_response == "London"
    assert s2.assertions is None


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_ground_truth(GroundTruthRef(path="/nope/gt.jsonl"))


def test_record_without_session_id_skipped(tmp_path):
    p = tmp_path / "gt.jsonl"
    p.write_text('{"expectedResponse": "orphan"}\n{"sessionId": "s1"}\n')
    gt = load_ground_truth(GroundTruthRef(path=str(p)))
    assert list(gt.by_session) == ["s1"]
