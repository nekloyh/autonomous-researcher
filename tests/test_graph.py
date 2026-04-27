"""Smoke tests for graph construction and routing logic."""
from __future__ import annotations

from app.graph import (
    after_critic,
    after_researcher,
    fan_out_or_synthesize,
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
        "findings": [],
        "draft_report": "",
        "critiques": [],
        "final_report": "",
        "citations": [],
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
    s = _state(
        critiques=[{"is_complete": True, "quality_score": 0.9, "missing_info": [], "factual_errors": [], "suggestions": []}],
        current_iteration=1,
    )
    assert after_critic(s) == "finalize"


def test_after_critic_replans_when_incomplete():
    s = _state(
        critiques=[{"is_complete": False, "quality_score": 0.4, "missing_info": ["What is X?"], "factual_errors": [], "suggestions": []}],
        current_iteration=1,
    )
    assert after_critic(s) == "replan"


def test_after_critic_finalizes_at_max_iterations():
    s = _state(
        critiques=[{"is_complete": False, "quality_score": 0.4, "missing_info": ["x"], "factual_errors": [], "suggestions": []}],
        current_iteration=3,
    )
    assert after_critic(s) == "finalize"


def test_build_graph_compiles():
    from app.graph import build_graph

    g = build_graph()
    assert g is not None
    # Inspect node names
    node_names = set(g.get_graph().nodes)
    for n in {"planner", "researcher", "synthesizer", "critic", "replan", "finalize"}:
        assert n in node_names, f"missing node: {n}"
