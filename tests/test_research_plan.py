"""Tests for general structured research planning and coverage."""
from __future__ import annotations

from datetime import datetime

from app.agents.planner import ResearchCellPlan, ResearchDimensionPlan, ResearchPlan, planner_node
from app.research_plan import build_cell_coverage, gap_to_cell, underfilled_cells


def _state(**overrides):
    base = {
        "user_query": "Compare VNG and FPT AI strategy in 2024",
        "session_id": "s",
        "started_at": datetime.now(),
        "plan": [],
        "research_plan": {},
        "cell_coverage": [],
        "current_iteration": 0,
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


def test_planner_node_converts_research_cells_to_subtasks(monkeypatch):
    monkeypatch.setattr("app.agents.planner.is_development", lambda: False)

    plan = ResearchPlan(
        reasoning="Strategy comparison requires entity-specific evidence cells.",
        query_intent="comparison",
        entities=["VNG", "FPT"],
        research_dimensions=[
            ResearchDimensionPlan(name="AI products", description="Products and services")
        ],
        research_cells=[
            ResearchCellPlan(
                id="cell_fpt_products",
                entity="FPT",
                dimension="AI products",
                question="What AI products did FPT offer in 2024?",
                target_queries=["FPT AI products 2024"],
                required_evidence=2,
                success_criteria=["Name concrete products or services."],
                evidence_type="product",
            )
        ],
        synthesis_requirements=["Compare each filled cell directly."],
    )
    monkeypatch.setattr("app.agents.planner._invoke_planner", lambda prompt: (plan, 123))

    out = planner_node(_state())

    assert out["research_plan"]["query_intent"] == "comparison"
    assert out["plan"][0]["cell_id"] == "cell_fpt_products"
    assert out["plan"][0]["target_queries"] == ["FPT AI products 2024"]
    assert out["plan"][0]["required_evidence"] == 2


def test_build_cell_coverage_and_gap_mapping():
    research_plan = {
        "research_cells": [
            {
                "id": "cell_fpt_partnerships",
                "entity": "FPT",
                "dimension": "partnerships",
                "question": "What AI partnerships did FPT announce in 2024?",
                "required_evidence": 2,
                "target_queries": ["FPT AI partnerships 2024"],
            }
        ]
    }
    findings = [
        {
            "task_id": "task_1",
            "claims": [
                {
                    "statement": "FPT announced an AI partnership.",
                    "source_url": "https://fpt.com/news",
                    "cell_id": "cell_fpt_partnerships",
                    "validation_status": "valid",
                }
            ],
        }
    ]

    coverage = build_cell_coverage(research_plan, findings)
    assert coverage[0]["validated_claims"] == 1
    assert coverage[0]["cell_status"] == "partial"
    assert underfilled_cells(research_plan, findings)[0]["id"] == "cell_fpt_partnerships"
    mapped = gap_to_cell(
        {"question": "Find FPT AI partnerships in 2024", "reason": "missing partnerships"},
        research_plan,
    )
    assert mapped["id"] == "cell_fpt_partnerships"
