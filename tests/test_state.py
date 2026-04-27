"""State reducer behavior."""
from operator import add


def test_findings_reducer_appends():
    a = [{"task_id": "t1", "content": "A"}]
    b = [{"task_id": "t2", "content": "B"}]
    merged = add(a, b)
    assert len(merged) == 2
    assert merged[0]["task_id"] == "t1"
    assert merged[1]["task_id"] == "t2"


def test_critiques_reducer_preserves_order():
    a = [{"is_complete": False, "quality_score": 0.6}]
    b = [{"is_complete": True, "quality_score": 0.85}]
    merged = add(a, b)
    assert merged[-1]["is_complete"] is True
