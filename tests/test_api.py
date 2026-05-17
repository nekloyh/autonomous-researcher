"""FastAPI smoke tests with mocked graph."""
from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.server import _summarize, app


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_research_endpoint_returns_report():
    fake_final = {
        "final_report": "# Test\n\nA report [1].\n\n## Sources\n[1] https://x",
        "citations": ["https://x"],
        "current_iteration": 1,
        "total_tool_calls": 4,
    }

    class FakeGraph:
        async def ainvoke(self, state, config):  # noqa: ARG002
            return fake_final

    with patch("app.api.server.get_graph", return_value=FakeGraph()):
        client = TestClient(app)
        r = client.post("/research", json={"query": "test query"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["final_report"].startswith("# Test")
        assert body["citations"] == ["https://x"]
        assert body["iterations"] == 1


def test_stream_summary_includes_live_ui_fields():
    planner = _summarize(
        "planner",
        {
            "plan": [
                {
                    "id": "task_1",
                    "question": "Find X",
                    "rationale": "Needed",
                    "dependencies": ["task_0"],
                }
            ],
            "current_iteration": 2,
            "total_tokens_used": 123,
        },
    )
    assert planner["tasks"] == [
        {
            "id": "task_1",
            "question": "Find X",
            "rationale": "Needed",
            "dependencies": ["task_0"],
        }
    ]
    assert planner["tokens"] == 123

    researcher = _summarize(
        "researcher",
        {
            "findings": [
                {
                    "task_id": "task_1",
                    "content": "A sourced finding.",
                    "sources": ["https://example.com"],
                    "claims": [{"statement": "x"}],
                    "confidence": 0.8,
                    "tool_calls": 3,
                }
            ],
            "total_tokens_used": 456,
        },
    )
    assert researcher["task_id"] == "task_1"
    assert researcher["excerpt"] == "A sourced finding."
    assert researcher["tool_calls"] == 3
    assert researcher["claims_count"] == 1
    assert researcher["tokens"] == 456
