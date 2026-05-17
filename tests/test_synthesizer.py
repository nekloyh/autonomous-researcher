"""Tests for the deterministic citation map / renumber pipeline."""
from __future__ import annotations

from app.agents.synthesizer import (
    _append_sources_section,
    _build_citation_map,
    _renumber_and_filter,
    synthesizer_node,
)


def _finding(task_id: str, claims: list[dict], sources: list[str] | None = None) -> dict:
    return {
        "task_id": task_id,
        "content": f"narrative for {task_id}",
        "claims": claims,
        "sources": sources or [c["source_url"] for c in claims],
        "confidence": 0.8,
        "tool_calls": 2,
    }


def test_citation_map_dedupes_and_orders():
    findings = [
        _finding(
            "t1",
            [
                {"statement": "x", "source_url": "https://a.example", "snippet": "s1", "confidence": 0.9},
                {"statement": "y", "source_url": "https://b.example", "snippet": "s2", "confidence": 0.8},
            ],
        ),
        _finding(
            "t2",
            [
                # duplicate URL must reuse same number
                {"statement": "z", "source_url": "https://a.example", "snippet": "s3", "confidence": 0.7},
                {"statement": "w", "source_url": "https://c.example", "snippet": "s4", "confidence": 0.7},
            ],
        ),
    ]
    cmap = _build_citation_map(findings)
    assert cmap["https://a.example"] == 1
    assert cmap["https://b.example"] == 2
    assert cmap["https://c.example"] == 3
    assert len(cmap) == 3


def test_renumber_drops_orphan_brackets():
    cmap = {"https://a.example": 1, "https://b.example": 2}
    draft = (
        "Claim one [1]. Claim two [2]. Hallucinated [7] should disappear. Repeat [1]."
    )
    out, new_cmap = _renumber_and_filter(draft, cmap)
    assert "[7]" not in out
    assert "[1]" in out and "[2]" in out
    # Original numbering happens to already be 1,2 so renumber is a no-op here.
    assert new_cmap["https://a.example"] == 1
    assert new_cmap["https://b.example"] == 2


def test_renumber_compacts_after_drop():
    cmap = {"https://a.example": 1, "https://b.example": 2, "https://c.example": 3}
    # Draft only references 2 and 3 → after renumber they should become 1 and 2.
    draft = "First fact [2]. Second fact [3]. Unsupported [9]."
    out, new_cmap = _renumber_and_filter(draft, cmap)
    assert "[9]" not in out
    assert new_cmap == {"https://b.example": 1, "https://c.example": 2}


def test_sources_section_appended_at_end():
    cmap = {"https://a.example": 1, "https://b.example": 2}
    body = "# Title\n\n## 1. Section\nClaim [1]. Other [2].\n"
    out = _append_sources_section(body, cmap)
    assert "## Sources" in out
    # Sources must be in numeric order
    a_idx = out.index("[1] https://a.example")
    b_idx = out.index("[2] https://b.example")
    assert a_idx < b_idx


def test_sources_section_replaces_existing():
    cmap = {"https://real.example": 1}
    body = (
        "# Title\n\n## 1. Section\nClaim [1].\n\n## Sources\n[1] https://fake.example\n"
    )
    out = _append_sources_section(body, cmap)
    assert "https://fake.example" not in out
    assert "https://real.example" in out


def test_citation_map_ignores_sources_without_claims():
    findings = [
        {
            "task_id": "t1",
            "content": "Narrative-only finding with no sourced claims.",
            "claims": [],
            "sources": ["https://a.example"],
            "confidence": 0.8,
            "tool_calls": 1,
        }
    ]
    assert _build_citation_map(findings) == {}


def test_synthesizer_does_not_use_unsupported_narrative_claim(monkeypatch):
    monkeypatch.setattr("app.agents.synthesizer.is_development", lambda: False)
    state = {
        "user_query": "MoMo user count",
        "session_id": "test_unsupported_claim",
        "findings": [
            {
                "task_id": "t1",
                "content": "MoMo has 50 million users.",
                "claims": [],
                "sources": ["https://momo.example"],
                "confidence": 0.8,
                "tool_calls": 1,
            }
        ],
    }
    out = synthesizer_node(state)
    assert out["citations"] == []
    assert "50 million" not in out["draft_report"]
    assert "not found in available sources" in out["draft_report"]


def test_synthesizer_drops_unrelated_vng_claim_for_momo(monkeypatch):
    monkeypatch.setattr("app.agents.synthesizer.is_development", lambda: False)
    state = {
        "user_query": "MoMo business model",
        "session_id": "test_momo_vng",
        "findings": [
            {
                "task_id": "t1",
                "content": "VNG unrelated context.",
                "claims": [
                    {
                        "statement": "VNG operates online games and digital services.",
                        "source_url": "https://vng.com.vn/about",
                        "snippet": "VNG operates online games and digital services.",
                        "confidence": 0.9,
                    }
                ],
                "sources": ["https://vng.com.vn/about"],
                "confidence": 0.8,
                "tool_calls": 1,
            }
        ],
    }
    out = synthesizer_node(state)
    assert "MoMo is part of VNG" not in out["draft_report"]
    assert "VNG operates" not in out["draft_report"]
    assert out["citations"] == []


def test_synthesizer_keeps_each_entity_in_comparison_query(monkeypatch):
    monkeypatch.setattr("app.agents.synthesizer.is_development", lambda: False)
    state = {
        "user_query": "MoMo vs ZaloPay vs VNPay market share",
        "session_id": "test_comparison_entities",
        "findings": [
            {
                "task_id": "t1",
                "content": "",
                "claims": [
                    {
                        "statement": "ZaloPay offers digital payment services.",
                        "source_url": "https://zalopay.vn/about",
                        "snippet": "ZaloPay offers digital payment services.",
                        "confidence": 0.9,
                    },
                    {
                        "statement": "VNPay offers merchant payment services.",
                        "source_url": "https://vnpay.vn/about",
                        "snippet": "VNPay offers merchant payment services.",
                        "confidence": 0.9,
                    },
                ],
                "sources": ["https://zalopay.vn/about", "https://vnpay.vn/about"],
                "confidence": 0.8,
                "tool_calls": 1,
            }
        ],
    }

    captured = {}

    class FakeLLM:
        def invoke(self, prompt, config=None):  # noqa: ARG002
            captured["prompt"] = prompt

            class Response:
                content = "# Report\n\nZaloPay is covered [1]. VNPay is covered [2]."

            return Response()

    monkeypatch.setattr("app.agents.synthesizer.get_synthesizer_llm", lambda: FakeLLM())
    out = synthesizer_node(state)

    assert "ZaloPay offers digital payment services." in captured["prompt"]
    assert "VNPay offers merchant payment services." in captured["prompt"]
    assert out["citations"] == ["https://zalopay.vn/about", "https://vnpay.vn/about"]


def test_synthesizer_uses_research_plan_coverage(monkeypatch):
    monkeypatch.setattr("app.agents.synthesizer.is_development", lambda: False)
    state = {
        "user_query": "Compare VNG and FPT AI strategy in 2024",
        "session_id": "test_plan_coverage",
        "research_plan": {
            "query_intent": "comparison",
            "synthesis_requirements": ["Compare filled cells and disclose missing cells."],
            "research_cells": [
                {
                    "id": "cell_fpt_partnerships",
                    "entity": "FPT",
                    "dimension": "partnerships",
                    "question": "What AI partnerships did FPT announce in 2024?",
                    "required_evidence": 2,
                    "target_queries": ["FPT AI partnerships 2024"],
                }
            ],
        },
        "findings": [
            {
                "task_id": "task_1",
                "cell_id": "cell_fpt_partnerships",
                "content": "",
                "claims": [
                    {
                        "statement": "FPT announced an AI partnership in 2024.",
                        "source_url": "https://fpt.com/news",
                        "snippet": "FPT announced an AI partnership in 2024.",
                        "confidence": 0.9,
                        "cell_id": "cell_fpt_partnerships",
                        "entity": "FPT",
                        "dimension": "partnerships",
                        "validation_status": "valid",
                    }
                ],
                "sources": ["https://fpt.com/news"],
                "confidence": 0.8,
                "tool_calls": 1,
            }
        ],
    }
    captured = {}

    class FakeLLM:
        def invoke(self, prompt, config=None):  # noqa: ARG002
            captured["prompt"] = prompt

            class Response:
                content = "# Report\n\nFPT has one partnership claim [1]."

            return Response()

    monkeypatch.setattr("app.agents.synthesizer.get_synthesizer_llm", lambda: FakeLLM())
    out = synthesizer_node(state)

    assert "Research plan coverage" in captured["prompt"]
    assert "Insufficient verified data after targeted research" in captured["prompt"]
    assert "cell_fpt_partnerships" in captured["prompt"]
    assert out["citations"] == ["https://fpt.com/news"]
