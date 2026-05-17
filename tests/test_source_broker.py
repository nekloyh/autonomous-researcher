"""Tests for deterministic source selection before researcher fan-out."""
from __future__ import annotations

from datetime import datetime

from app.agents.source_broker import (
    canonicalize_url,
    classify_source,
    rank_source,
    source_broker_node,
)


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


def test_source_broker_keeps_entity_specific_comparison_queries(monkeypatch):
    monkeypatch.setattr("app.agents.source_broker.is_development", lambda: False)
    monkeypatch.setattr("app.agents.source_broker.seed_source_results", lambda *args, **kwargs: [])
    seen_queries = []

    def fake_search(query, max_results=8):  # noqa: ARG001
        seen_queries.append(query)
        return [
            {
                "title": "VNG annual report",
                "url": "https://bctn2024.vng.com.vn/about/people-strategy",
                "content": "VNG AI strategy",
            }
        ]

    monkeypatch.setattr("app.agents.source_broker.search_web_results", fake_search)
    source_broker_node(
        _state(
            user_query="Compare VNG and FPT AI strategy in 2024",
            plan=[
                {
                    "id": "task_1",
                    "question": "What is VNG Corporation's AI strategy in 2024?",
                    "rationale": "R",
                    "dependencies": [],
                    "status": "pending",
                }
            ],
        )
    )
    assert seen_queries == ["What is VNG Corporation's AI strategy in 2024?"]


def test_rank_source_penalizes_wrong_year_when_query_has_year():
    right_year = rank_source(
        "FPT AI strategy 2024",
        {
            "title": "FPT unveils strategic directions in 2024",
            "url": "https://fptsoftware.com/news/2024/ai",
            "content": "FPT AI strategy in 2024",
        },
    )
    wrong_year = rank_source(
        "FPT AI strategy 2024",
        {
            "title": "FPT launches AI-first platform in 2025",
            "url": "https://fptsoftware.com/news/2025/ai",
            "content": "FPT AI strategy update in 2025",
        },
    )

    assert right_year > wrong_year


def test_source_broker_blocks_low_quality_domains(monkeypatch):
    monkeypatch.setattr("app.agents.source_broker.is_development", lambda: False)

    def fake_search(query, max_results=8):  # noqa: ARG001
        return [
            {
                "title": "VNG AI strategy mirror",
                "url": "https://scribd.com/document/123/vng-ai",
                "content": "VNG AI strategy in 2024",
            },
            {
                "title": "FPT social update",
                "url": "https://facebook.com/fpt/posts/1",
                "content": "FPT AI strategy in 2024",
            },
        ]

    monkeypatch.setattr("app.agents.source_broker.search_web_results", fake_search)
    out = source_broker_node(
        _state(
            user_query="Compare VNG and FPT AI strategy in 2024",
            plan=[
                {
                    "id": "task_1",
                    "question": "What is VNG AI strategy in 2024?",
                    "rationale": "R",
                    "dependencies": [],
                    "status": "pending",
                }
            ],
        )
    )

    domains = {c["domain"] for c in out["source_candidates"]}
    assert "scribd.com" not in domains
    assert "facebook.com" not in domains
    assert all(c["source_policy_tier"] != "blocked" for c in out["source_candidates"])


def test_source_broker_seeds_known_entity_sources(monkeypatch):
    monkeypatch.setattr("app.agents.source_broker.is_development", lambda: False)
    monkeypatch.setattr("app.agents.source_broker.search_web_results", lambda *args, **kwargs: [])

    out = source_broker_node(
        _state(
            user_query="Compare VNG and FPT AI strategy in 2024",
            plan=[
                {
                    "id": "task_1",
                    "question": "What is FPT Corporation's AI strategy in 2024?",
                    "rationale": "R",
                    "dependencies": [],
                    "status": "pending",
                }
            ],
        )
    )

    urls = {c["canonical_url"] for c in out["source_candidates"]}
    assert any("fptsoftware.com" in url for url in urls)
    assert all(c["source_policy_tier"] == "preferred" for c in out["source_candidates"])


def test_source_broker_uses_cell_target_queries(monkeypatch):
    monkeypatch.setattr("app.agents.source_broker.is_development", lambda: False)
    monkeypatch.setattr("app.agents.source_broker.seed_source_results", lambda *args, **kwargs: [])
    seen_queries = []

    def fake_search(query, max_results=8):  # noqa: ARG001
        seen_queries.append(query)
        return [
            {
                "title": "FPT AI partnership",
                "url": "https://fpt.com/en/news/fpt-news/ai-partnership-2024",
                "content": "FPT announced an AI partnership in 2024.",
            }
        ]

    monkeypatch.setattr("app.agents.source_broker.search_web_results", fake_search)
    out = source_broker_node(
        _state(
            user_query="Compare VNG and FPT AI strategy in 2024",
            plan=[
                {
                    "id": "task_1",
                    "cell_id": "cell_fpt_partnerships",
                    "entity": "FPT",
                    "dimension": "partnerships",
                    "question": "What AI partnerships did FPT announce in 2024?",
                    "target_queries": ["FPT AI partnerships 2024"],
                    "required_evidence": 2,
                    "success_criteria": [],
                    "allow_insufficient_data": False,
                    "rationale": "R",
                    "dependencies": [],
                    "status": "pending",
                }
            ],
        )
    )

    assert seen_queries == ["FPT AI partnerships 2024"]
    assert out["source_candidates"][0]["assigned_cell_ids"] == ["cell_fpt_partnerships"]
