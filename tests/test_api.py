"""FastAPI smoke tests with mocked graph."""
from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.server import app


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
