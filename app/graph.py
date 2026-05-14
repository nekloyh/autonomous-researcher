"""LangGraph orchestration: Planner → fan-out Researchers → Synthesizer → Critic loop."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents import (
    critic_node,
    planner_node,
    researcher_node,
    should_continue,
    source_broker_node,
    synthesizer_node,
)
from app.config import MAX_GAP_ROUNDS, MAX_PARALLEL
from app.entity_guard import entities_in_text, is_entity_contaminated
from app.evaluation.cost import estimate_cost_simple
from app.logger import log_event
from app.state import AgentState

MIN_FINAL_CITATIONS = 3
MIN_SOURCED_CLAIMS = 3
MIN_CRITIC_SCORE = 0.65
OUTPUT_DIR = Path("outputs")


def _ready_tasks(state: AgentState) -> list:
    done_ids = {f.get("task_id") for f in state.get("findings", []) if f.get("task_id")}
    ready = []
    for t in state.get("plan", []):
        if t["id"] in done_ids:
            continue
        if all(dep in done_ids for dep in t.get("dependencies", [])):
            ready.append(t)
    return ready


def _sources_for_task(state: AgentState, task_id: str) -> list[dict]:
    sources = []
    seen = set()
    for source in state.get("source_candidates", []) or []:
        if task_id not in (source.get("assigned_task_ids") or []):
            continue
        canonical = source.get("canonical_url") or source.get("url")
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        sources.append(source)
    sources.sort(key=lambda s: float(s.get("rank_score", 0.0)), reverse=True)
    return sources


def fan_out_or_synthesize(state: AgentState):
    """After planner (or replan), dispatch parallel researchers, or skip to synthesis."""
    ready = _ready_tasks(state)
    if not ready:
        return "synthesizer"
    batch = ready[:MAX_PARALLEL]
    return [
        Send(
            "researcher",
            {
                "task": t,
                "user_query": state["user_query"],
                "session_id": state.get("session_id", ""),
                "assigned_sources": _sources_for_task(state, t["id"]),
            },
        )
        for t in batch
    ]


def after_researcher(state: AgentState):
    """Send another researcher batch if there are pending tasks; else synthesize."""
    ready = _ready_tasks(state)
    if not ready:
        return "synthesizer"
    return "fan_out"


def fan_out_node(state: AgentState) -> dict:
    """Barrier node before dispatching the next researcher batch."""
    return {}


def after_critic(state: AgentState) -> Literal["research_gaps", "replan", "finalize"]:
    quality = _evaluate_quality_gate(state)
    if quality["hard_stop"]:
        return "finalize"

    critiques = state.get("critiques", []) or []
    last = critiques[-1] if critiques else {}
    action = last.get("action")
    if action not in {"finalize", "research_gaps", "replan"}:
        decision = should_continue(state)
        action = "replan" if decision == "replan" else "finalize"

    if action == "replan":
        return "replan"

    if action == "research_gaps":
        if state.get("gap_rounds", 0) >= MAX_GAP_ROUNDS:
            return "replan"
        if _critique_gap_questions(last):
            return "research_gaps"
        return "replan"

    if quality["gate_passed"]:
        return "finalize"
    if state.get("gap_rounds", 0) < MAX_GAP_ROUNDS:
        return "research_gaps"
    return "replan"


def replan_node(state: AgentState) -> dict:
    """Re-run planner with the latest critique as feedback."""
    return planner_node(state)


def _critique_gap_questions(critique: dict) -> list[str]:
    questions = []
    for gap in critique.get("gaps") or []:
        question = (gap.get("question") or "").strip()
        if question:
            questions.append(question)
    for item in critique.get("missing_info") or []:
        question = str(item).strip()
        if question and question not in questions:
            questions.append(question)
    return questions


def gap_planner_node(state: AgentState) -> dict:
    """Convert critic gaps into targeted temporary subtasks."""
    critiques = state.get("critiques", []) or []
    last = critiques[-1] if critiques else {}
    questions = _critique_gap_questions(last)
    if not questions:
        questions = [str(w) for w in state.get("quality_warnings", []) or [] if str(w).strip()]
    subtasks = []
    iteration = state.get("current_iteration", 0)
    for i, question in enumerate(questions[:MAX_PARALLEL], 1):
        subtasks.append(
            {
                "id": f"gap_{iteration}_{state.get('gap_rounds', 0) + 1}_{i}",
                "question": question,
                "rationale": "Targeted follow-up from critic or quality gate.",
                "dependencies": [],
                "status": "pending",
            }
        )
    log_event(
        "gap_planner",
        state.get("session_id", "-"),
        gaps=len(subtasks),
        gap_rounds=state.get("gap_rounds", 0) + 1,
    )
    return {"plan": subtasks, "gap_rounds": state.get("gap_rounds", 0) + 1}


def quality_gate_node(state: AgentState) -> dict:
    """Add deterministic gate feedback before deciding whether to finalize."""
    quality = _evaluate_quality_gate(state)
    update: dict = {
        "quality_status": quality["status"],
        "quality_warnings": quality["warnings"],
    }
    if should_continue(state) == "finish" and not quality["gate_passed"] and not quality["hard_stop"]:
        update["critiques"] = [
            {
                "action": "research_gaps",
                "is_complete": False,
                "quality_score": float(quality["critic_score"] or 0.0),
                "missing_info": list(quality["warnings"]),
                "gaps": [
                    {
                        "question": warning,
                        "origin_task_id": "quality_gate",
                        "reason": "Deterministic quality gate failure.",
                        "priority": "high",
                    }
                    for warning in quality["warnings"]
                ],
                "factual_errors": list(quality["warnings"]),
                "unsupported_claims": [],
                "conflicting_claims": [],
                "suggestions": [
                    "Gather more source-backed claims and regenerate the report before finalizing."
                ],
            }
        ]
    log_event(
        "quality_gate",
        state.get("session_id", "-"),
        status=quality["status"],
        gate_passed=quality["gate_passed"],
        hard_stop=quality["hard_stop"],
        warnings=quality["warnings"],
    )
    return update


def _sourced_claims(state: AgentState) -> list[dict]:
    claims: list[dict] = []
    for finding in state.get("findings", []) or []:
        for claim in finding.get("claims") or []:
            if claim.get("source_url"):
                claims.append(claim)
    return claims


def _critic_provider_failures(state: AgentState) -> list[str]:
    failures: list[str] = []
    for err in state.get("errors", []) or []:
        lower = str(err).lower()
        if "critic" in lower and ("provider" in lower or "failure" in lower or "error" in lower):
            failures.append(str(err))
    return failures


def _citation_numbers(report: str) -> set[int]:
    numbers = set()
    for n in re.findall(r"\[(\d+)\]", report or ""):
        try:
            numbers.add(int(n))
        except ValueError:
            continue
    return numbers


def _evaluate_quality_gate(state: AgentState) -> dict:
    iteration = state.get("current_iteration", 0)
    max_iter = state.get("max_iterations") or 0
    hard_stop = bool(max_iter and iteration >= max_iter)
    citations = state.get("citations", []) or []
    claims = _sourced_claims(state)
    critiques = state.get("critiques", []) or []
    last_score = None
    if critiques:
        try:
            last_score = float(critiques[-1].get("quality_score", 0.0))
        except (TypeError, ValueError):
            last_score = 0.0

    failures: list[str] = []
    warnings: list[str] = []
    report_numbers = _citation_numbers(state.get("draft_report", ""))
    if report_numbers and citations:
        max_allowed = len(citations)
        orphans = sorted(n for n in report_numbers if n < 1 or n > max_allowed)
        if orphans:
            failures.append(f"orphan citation numbers in report: {orphans[:5]}")
    if len(citations) < MIN_FINAL_CITATIONS:
        failures.append(
            f"only {len(citations)} citations found; minimum is {MIN_FINAL_CITATIONS}"
        )
    if len(claims) < MIN_SOURCED_CLAIMS:
        failures.append(
            f"only {len(claims)} sourced claims found; minimum is {MIN_SOURCED_CLAIMS}"
        )
    if last_score is None:
        failures.append("missing critic review")
    elif last_score < MIN_CRITIC_SCORE:
        failures.append(
            f"critic score {last_score:.2f} is below minimum {MIN_CRITIC_SCORE:.2f}"
        )

    contaminated = [
        c.get("statement", "")[:80]
        for c in claims
        if is_entity_contaminated(
            state.get("user_query", ""),
            f"{c.get('statement', '')}\n{c.get('snippet', '')}",
            c.get("source_url", ""),
        )
    ]
    if contaminated:
        failures.append(f"entity contamination detected in {len(contaminated)} sourced claims")

    query_entities = entities_in_text(state.get("user_query", ""))
    if len(query_entities) >= 2 and re.search(r"\b(compare|versus|vs|difference|so sánh)\b", state.get("user_query", ""), re.I):
        coverage = {entity: 0 for entity in query_entities}
        for c in claims:
            mentioned = entities_in_text(f"{c.get('statement', '')}\n{c.get('snippet', '')}\n{c.get('source_url', '')}")
            for entity in query_entities & mentioned:
                coverage[entity] += 1
        missing_entities = [e for e, count in coverage.items() if count == 0]
        if missing_entities:
            failures.append(
                "comparison query lacks sourced claims for entities: "
                + ", ".join(sorted(missing_entities))
            )

    provider_failures = _critic_provider_failures(state)
    if provider_failures:
        failures.append("critic/provider failure was recorded")
        warnings.extend(provider_failures)

    warnings.extend(failures)
    gate_passed = not failures
    return {
        "gate_passed": gate_passed,
        "hard_stop": hard_stop,
        "status": "verified" if gate_passed else "unverified",
        "warnings": list(dict.fromkeys(warnings)),
        "citations_count": len(citations),
        "sourced_claims_count": len(claims),
        "critic_score": last_score,
    }


def _safe_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id or "session").strip("._")
    return cleaned or "session"


def _write_run_summary(state: AgentState, final_report: str, quality: dict) -> str:
    session_id = _safe_session_id(state.get("session_id", "session"))
    path = OUTPUT_DIR / f"{session_id}.json"
    claims = _sourced_claims(state)
    payload = {
        "query": state.get("user_query", ""),
        "plan": state.get("plan", []),
        "findings": state.get("findings", []),
        "source_candidates": state.get("source_candidates", []) or [],
        "claims": claims,
        "citations": state.get("citations", []) or [],
        "critiques": state.get("critiques", []) or [],
        "unresolved_gaps": [
            gap
            for critique in state.get("critiques", []) or []
            for gap in (critique.get("gaps") or [])
        ],
        "errors": state.get("errors", []) or [],
        "final_quality_status": quality.get("status", "unverified"),
        "quality_warnings": quality.get("warnings", []),
        "final_report": final_report,
    }
    try:
        OUTPUT_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return str(path)
    except Exception as e:
        log_event(
            "finalize",
            state.get("session_id", "-"),
            summary_write_error=f"{type(e).__name__}: {e}",
        )
        return ""


def finalize_node(state: AgentState) -> dict:
    quality = _evaluate_quality_gate(state)
    iters = state.get("current_iteration", 1)
    tool_calls = state.get("total_tool_calls", 0)
    tokens = state.get("total_tokens_used", 0)
    started = state.get("started_at")
    duration = ""
    if isinstance(started, datetime):
        duration = f" • duration={(datetime.now() - started).total_seconds():.1f}s"
    cost_str = ""
    if tokens:
        cost = estimate_cost_simple(tokens)
        cost_str = f" • cost≈${cost:.4f}"
    tokens_str = f" • tokens≈{tokens}" if tokens else ""
    quality_str = f" • quality={quality['status']}"
    footer = (
        f"\n\n---\n*Generated by autonomous-researcher · iterations={iters} · "
        f"tool_calls={tool_calls}{tokens_str}{cost_str}{duration}{quality_str}*"
    )
    draft = state.get("draft_report", "") or ""
    if quality["status"] == "unverified":
        warning_text = "; ".join(quality["warnings"]) or "quality gate did not pass"
        draft = (
            draft.rstrip()
            + "\n\n> **Verification warning:** This report is unverified. "
            + warning_text
            + "\n"
        )
    final = draft + footer
    summary_path = _write_run_summary(state, final, quality)
    log_event(
        "finalize",
        state.get("session_id", "-"),
        verified=quality["status"] == "verified",
        status=quality["status"],
        citations=quality["citations_count"],
        sourced_claims=quality["sourced_claims_count"],
        critic_score=quality["critic_score"],
        warnings=quality["warnings"],
        summary_path=summary_path,
    )
    return {
        "final_report": final,
        "quality_status": quality["status"],
        "quality_warnings": quality["warnings"],
        "run_summary_path": summary_path,
    }


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("source_broker", source_broker_node)
    g.add_node("fan_out", fan_out_node)
    g.add_node("researcher", researcher_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("critic", critic_node)
    g.add_node("quality_gate", quality_gate_node)
    g.add_node("replan", replan_node)
    g.add_node("gap_planner", gap_planner_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "source_broker")
    g.add_edge("replan", "source_broker")
    g.add_edge("gap_planner", "source_broker")
    g.add_edge("source_broker", "fan_out")

    g.add_conditional_edges(
        "researcher",
        after_researcher,
        ["fan_out", "synthesizer"],
    )
    g.add_conditional_edges(
        "fan_out",
        fan_out_or_synthesize,
        ["researcher", "synthesizer"],
    )

    g.add_edge("synthesizer", "critic")
    g.add_edge("critic", "quality_gate")

    g.add_conditional_edges(
        "quality_gate",
        after_critic,
        {"research_gaps": "gap_planner", "replan": "replan", "finalize": "finalize"},
    )

    g.add_edge("finalize", END)

    if checkpointer is None:
        checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


_graph = None


def get_graph(checkpointer=None):
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=checkpointer)
    return _graph


def reset_graph() -> None:
    global _graph
    _graph = None
