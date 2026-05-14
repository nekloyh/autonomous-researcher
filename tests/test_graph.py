"""Smoke tests for graph construction and routing logic."""
from __future__ import annotations

from app.agents.critic import critic_node
from app.graph import (
    after_critic,
    after_researcher,
    fan_out_or_synthesize,
    finalize_node,
    gap_planner_node,
)
from app.state import AgentState


def _state(**overrides) -> AgentState:
    base: AgentState = {
        "user_query": "test",
        "session_id": "s",
        "started_at": None,  # type: ignore[typeddict-item]
        "plan": [],
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


def test_fan_out_skips_to_synth_when_plan_empty():
    s = _state(plan=[])
    assert fan_out_or_synthesize(s) == "synthesizer"


def test_fan_out_emits_sends_for_pending_tasks():
    plan = [
        {"id": "task_1", "question": "Q1", "rationale": "R", "dependencies": [], "status": "pending"},
        {"id": "task_2", "question": "Q2", "rationale": "R", "dependencies": [], "status": "pending"},
    ]
    s = _state(plan=plan)
    sends = fan_out_or_synthesize(s)
    assert isinstance(sends, list)
    assert all(getattr(send, "node", None) == "researcher" for send in sends)
    assert len(sends) == 2
    assert sends[0].arg["assigned_sources"] == []


def test_fan_out_attaches_assigned_sources():
    plan = [
        {"id": "task_1", "question": "Q1", "rationale": "R", "dependencies": [], "status": "pending"},
    ]
    s = _state(
        plan=plan,
        source_candidates=[
            {
                "url": "https://a.example",
                "canonical_url": "https://a.example",
                "title": "A",
                "snippet": "A",
                "domain": "a.example",
                "rank_score": 7.0,
                "source_type": "official",
                "assigned_task_ids": ["task_1"],
            }
        ],
    )
    sends = fan_out_or_synthesize(s)
    assert len(sends) == 1
    assert sends[0].arg["assigned_sources"][0]["url"] == "https://a.example"


def test_fan_out_respects_dependencies():
    plan = [
        {"id": "task_1", "question": "Q1", "rationale": "R", "dependencies": [], "status": "pending"},
        {"id": "task_2", "question": "Q2", "rationale": "R", "dependencies": ["task_1"], "status": "pending"},
    ]
    s = _state(plan=plan)
    sends = fan_out_or_synthesize(s)
    # Only task_1 is dispatchable until it finishes
    assert len(sends) == 1
    assert sends[0].arg["task"]["id"] == "task_1"


def test_after_researcher_loops_when_more_tasks_pending():
    plan = [
        {"id": "task_1", "question": "Q1", "rationale": "R", "dependencies": [], "status": "pending"},
        {"id": "task_2", "question": "Q2", "rationale": "R", "dependencies": [], "status": "pending"},
    ]
    s = _state(plan=plan, findings=[{"task_id": "task_1", "content": "x", "sources": [], "confidence": 0.9, "tool_calls": 1}])
    assert after_researcher(s) == "fan_out"


def test_after_researcher_proceeds_when_all_done():
    plan = [
        {"id": "task_1", "question": "Q1", "rationale": "R", "dependencies": [], "status": "pending"},
    ]
    s = _state(plan=plan, findings=[{"task_id": "task_1", "content": "x", "sources": [], "confidence": 0.9, "tool_calls": 1}])
    assert after_researcher(s) == "synthesizer"


def test_after_critic_finishes_when_complete():
    findings = [
        {
            "task_id": "t1",
            "content": "x",
            "sources": ["https://a.example", "https://b.example", "https://c.example"],
            "claims": [
                {"statement": "a", "source_url": "https://a.example", "snippet": "a", "confidence": 0.9},
                {"statement": "b", "source_url": "https://b.example", "snippet": "b", "confidence": 0.9},
                {"statement": "c", "source_url": "https://c.example", "snippet": "c", "confidence": 0.9},
            ],
            "confidence": 0.9,
            "tool_calls": 1,
        }
    ]
    s = _state(
        critiques=[{"is_complete": True, "quality_score": 0.9, "missing_info": [], "factual_errors": [], "suggestions": []}],
        current_iteration=1,
        citations=["https://a.example", "https://b.example", "https://c.example"],
        findings=findings,
    )
    assert after_critic(s) == "finalize"


def test_after_critic_replans_when_incomplete():
    s = _state(
        critiques=[{"action": "replan", "is_complete": False, "quality_score": 0.4, "missing_info": ["What is X?"], "factual_errors": [], "suggestions": []}],
        current_iteration=1,
    )
    assert after_critic(s) == "replan"


def test_after_critic_researches_targeted_gaps_when_partial():
    s = _state(
        critiques=[
            {
                "action": "research_gaps",
                "is_complete": False,
                "quality_score": 0.7,
                "missing_info": ["What is X?"],
                "gaps": [{"question": "What is X?", "origin_task_id": "task_1", "reason": "missing", "priority": "high"}],
                "factual_errors": [],
                "suggestions": [],
            }
        ],
        current_iteration=1,
    )
    assert after_critic(s) == "research_gaps"


def test_after_critic_finalizes_at_max_iterations():
    s = _state(
        critiques=[{"is_complete": False, "quality_score": 0.4, "missing_info": ["x"], "factual_errors": [], "suggestions": []}],
        current_iteration=3,
    )
    assert after_critic(s) == "finalize"


def test_after_critic_replans_when_quality_gate_fails_before_max():
    s = _state(
        critiques=[{"is_complete": True, "quality_score": 0.9, "missing_info": [], "factual_errors": [], "suggestions": []}],
        current_iteration=1,
        citations=["https://a.example"],
        findings=[
            {
                "task_id": "t1",
                "content": "x",
                "sources": ["https://a.example"],
                "claims": [
                    {"statement": "a", "source_url": "https://a.example", "snippet": "a", "confidence": 0.9}
                ],
                "confidence": 0.9,
                "tool_calls": 1,
            }
        ],
    )
    assert after_critic(s) == "research_gaps"


def test_gap_planner_creates_temporary_gap_tasks():
    s = _state(
        current_iteration=2,
        critiques=[
            {
                "action": "research_gaps",
                "is_complete": False,
                "quality_score": 0.7,
                "missing_info": [],
                "gaps": [{"question": "Find launch date", "origin_task_id": "task_1", "reason": "missing", "priority": "high"}],
                "factual_errors": [],
                "suggestions": [],
            }
        ],
    )
    out = gap_planner_node(s)
    assert out["gap_rounds"] == 1
    assert out["plan"][0]["id"].startswith("gap_2_1_")
    assert out["plan"][0]["question"] == "Find launch date"


def test_critic_provider_failure_replans_before_max(monkeypatch):
    class BrokenLLM:
        def with_structured_output(self, *args, **kwargs):  # noqa: ARG002
            return self

        def invoke(self, *args, **kwargs):  # noqa: ARG002
            raise RuntimeError("provider down")

    monkeypatch.setattr("app.agents.critic.is_development", lambda: False)
    monkeypatch.setattr("app.agents.critic.get_critic_llm", lambda: BrokenLLM())

    s = _state(current_iteration=1, max_iterations=3, draft_report="# Draft")
    update = critic_node(s)
    critique = update["critiques"][0]
    assert critique["quality_score"] == 0.0
    assert critique["is_complete"] is False
    assert "critic/provider failure" in update["errors"][0]

    routed = _state(
        current_iteration=1,
        max_iterations=3,
        critiques=update["critiques"],
        errors=update["errors"],
    )
    assert after_critic(routed) == "replan"


def test_critic_provider_failure_finalizes_unverified_at_max(monkeypatch):
    class BrokenLLM:
        def with_structured_output(self, *args, **kwargs):  # noqa: ARG002
            return self

        def invoke(self, *args, **kwargs):  # noqa: ARG002
            raise RuntimeError("provider down")

    monkeypatch.setattr("app.agents.critic.is_development", lambda: False)
    monkeypatch.setattr("app.agents.critic.get_critic_llm", lambda: BrokenLLM())

    s = _state(current_iteration=3, max_iterations=3, draft_report="# Draft")
    update = critic_node(s)
    critique = update["critiques"][0]
    assert critique["quality_score"] == 0.0
    assert critique["is_complete"] is True

    final_state = _state(
        current_iteration=3,
        max_iterations=3,
        draft_report="# Draft",
        critiques=update["critiques"],
        errors=update["errors"],
    )
    out = finalize_node(final_state)
    assert out["quality_status"] == "unverified"
    assert "Verification warning" in out["final_report"]
    assert "critic/provider failure" in out["final_report"]


def test_build_graph_compiles():
    from app.graph import build_graph

    g = build_graph()
    assert g is not None
    # Inspect node names
    node_names = set(g.get_graph().nodes)
    for n in {"planner", "source_broker", "fan_out", "researcher", "synthesizer", "critic", "quality_gate", "gap_planner", "replan", "finalize"}:
        assert n in node_names, f"missing node: {n}"
