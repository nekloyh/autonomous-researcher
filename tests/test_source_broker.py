"""Tests for deterministic source selection before researcher fan-out."""
from __future__ import annotations

from datetime import datetime

from app.agents.source_broker import canonicalize_url, classify_source, source_broker_node


def _state(**overrides):
    base = {
        "user_query": "MoMo business model",
        "session_id": "s",
        "started_at": datetime.now(),
        "plan": [
            {"id": "task_1", "question": "MoMo business model", "rationale": "R", "dependencies": [], "status": "pending"},
            {"id": "task_2", "question": "MoMo revenue sources", "rationale": "R", "dependencies": [], "status": "pending"},
        ],
        "current_iteration": 1,
        "max_iterations": 3,
        "gap_rounds": 0,
        "findings": [],
        "source_candidates": [],
        "draft_report": "",
        "critiques": [],
        "final_report": "",
        "citations": [],
        "quality_status": "unverified",
        "quality_warnings": [],
        "run_summary_path": "",
        "total_tool_calls": 0,
        "total_tokens_used": 0,
        "errors": [],
    }
    base.update(overrides)
    return base


def test_canonicalize_url_strips_tracking_params():
    out = canonicalize_url("https://www.Example.com/path/?utm_source=x&keep=1&fbclid=y")
    assert out == "https://example.com/path?keep=1"


def test_classify_source_prefers_entity_domain_as_official():
    assert classify_source("https://momo.vn/news", "MoMo update", "", "MoMo business model") == "official"


def test_source_broker_ranks_and_dedupes(monkeypatch):
    monkeypatch.setattr("app.agents.source_broker.is_development", lambda: False)

    def fake_search(query, max_results=8):  # noqa: ARG001
        return [
            {"title": "Generic", "url": "https://medium.com/x?utm_source=a", "content": "MoMo business model"},
            {"title": "MoMo official press release", "url": "https://www.momo.vn/news/update?utm_campaign=a", "content": "official press release"},
            {"title": "MoMo duplicate", "url": "https://momo.vn/news/update", "content": "same"},
        ]

    monkeypatch.setattr("app.agents.source_broker.search_web_results", fake_search)
    out = source_broker_node(_state())
    candidates = out["source_candidates"]
    canonicals = [c["canonical_url"] for c in candidates]
    assert canonicals.count("https://momo.vn/news/update") == 1
    assert candidates[0]["source_type"] == "official"
    assert set(candidates[0]["assigned_task_ids"]) == {"task_1", "task_2"}
