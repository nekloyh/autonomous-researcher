"""Planner agent: decomposes user query into independent sub-tasks."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import MAX_SUBTASKS, get_planner_llm, is_development
from app.logger import log_event
from app.observability import tokens_from_response
from app.prompts import PLANNER_PROMPT
from app.prompts.planner import PROMPT_VERSION
from app.provider_rotation import invoke_with_rotation, rotate_groq
from app.state import AgentState, SubTask


class SubTaskPlan(BaseModel):
    id: str = Field(description="Unique identifier like 'task_1'")
    question: str = Field(description="Specific, researchable question")
    rationale: str = Field(description="Why this task is needed to answer the main query")
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of tasks that must complete before this one (rare, usually empty)",
    )


class ResearchDimensionPlan(BaseModel):
    name: str = Field(description="Dimension derived from the user query.")
    description: str = Field(default="", description="What this dimension covers.")


class ResearchCellPlan(BaseModel):
    id: str = Field(description="Stable cell id like cell_1")
    entity: str = Field(default="", description="Entity being researched; empty if not entity-specific.")
    dimension: str = Field(description="Research dimension this cell covers.")
    question: str = Field(description="Specific, answerable question for this cell.")
    target_queries: list[str] = Field(default_factory=list)
    required_evidence: int = Field(default=1, ge=1, le=5)
    success_criteria: list[str] = Field(default_factory=list)
    evidence_type: str = Field(default="factual")
    allow_insufficient_data: bool = Field(default=False)


class ResearchPlan(BaseModel):
    reasoning: str = Field(description="Brief explanation of the decomposition strategy")
    query_intent: Literal["comparison", "analysis", "factual", "numerical", "exploratory", "multi_hop"] = "exploratory"
    entities: list[str] = Field(default_factory=list)
    research_dimensions: list[ResearchDimensionPlan] = Field(default_factory=list)
    research_cells: list[ResearchCellPlan] = Field(default_factory=list)
    synthesis_requirements: list[str] = Field(default_factory=list)
    tasks: list[SubTaskPlan] = Field(default_factory=list)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8), reraise=True)
def _invoke_planner(prompt: str) -> tuple[ResearchPlan, int]:
    def _call():
        llm = get_planner_llm().with_structured_output(ResearchPlan, include_raw=True)
        return llm.invoke(
            prompt,
            config={"metadata": {"prompt_version": PROMPT_VERSION, "agent": "planner"}},
        )

    out = invoke_with_rotation(
        "groq",
        _call,
        attempts=2,
        rotate=rotate_groq,
    )
    if isinstance(out, dict):
        if out.get("parsing_error") and not out.get("parsed"):
            raise out["parsing_error"]
        parsed = out.get("parsed")
        tokens = tokens_from_response(out.get("raw"))
    else:
        parsed = out
        tokens = 0
    return parsed, tokens


def planner_node(state: AgentState) -> dict:
    """Planning node — runs at start and on each replan."""
    iteration = state.get("current_iteration", 0)

    if is_development():
        query = state["user_query"]
        plan: list[SubTask] = [
            {
                "id": "task_1",
                "question": f"Development-mode summary for: {query}",
                "rationale": "Deterministic local stub; no external LLM calls.",
                "dependencies": [],
                "status": "pending",
                "cell_id": "cell_dev_1",
                "entity": "",
                "dimension": "development smoke",
                "target_queries": [query],
                "required_evidence": 1,
                "success_criteria": ["Development smoke produces a final report."],
                "allow_insufficient_data": True,
            }
        ]
        log_event(
            "planner",
            state.get("session_id", "-"),
            iteration=iteration + 1,
            subtasks=len(plan),
            development=True,
        )
        return {
            "plan": plan,
            "research_plan": {
                "reasoning": "Development-mode deterministic planner.",
                "query_intent": "exploratory",
                "entities": [],
                "research_dimensions": [
                    {"name": "development smoke", "description": "Local deterministic run."}
                ],
                "research_cells": [
                    {
                        "id": "cell_dev_1",
                        "task_id": "task_1",
                        "entity": "",
                        "dimension": "development smoke",
                        "question": plan[0]["question"],
                        "target_queries": [query],
                        "required_evidence": 1,
                        "success_criteria": ["Development smoke produces a final report."],
                        "evidence_type": "factual",
                        "allow_insufficient_data": True,
                    }
                ],
                "synthesis_requirements": ["Write a concise development report."],
            },
            "current_iteration": iteration + 1,
            "gap_rounds": 0,
        }

    previous_context = ""
    if state.get("critiques"):
        last = state["critiques"][-1]
        gaps = "\n".join(f"- {m}" for m in last.get("missing_info", []))
        if gaps:
            previous_context = (
                "The previous draft was incomplete. Focus the new plan on filling "
                "ONLY these gaps; do not redo what is already covered:\n" + gaps
            )

    existing_coverage = "(no prior context)"
    if iteration > 0 and state.get("findings"):
        lines = []
        for f in state["findings"]:
            tid = f.get("task_id", "?")
            summary = (f.get("content") or "").strip().replace("\n", " ")
            lines.append(f"- {tid}: {summary[:120]}")
        existing_coverage = (
            "Already-answered sub-tasks (DO NOT replan these; only fill gaps):\n"
            + "\n".join(lines)
        )

    prompt = PLANNER_PROMPT.format(
        query=state["user_query"],
        previous_context=previous_context or "(none — this is the first iteration)",
        known_context=existing_coverage,
        max_tasks=MAX_SUBTASKS,
    )

    plan, tokens = _invoke_planner(prompt)

    suffix = f"_iter{iteration + 1}" if iteration > 0 else ""
    cells = plan.research_cells[:MAX_SUBTASKS]
    if cells:
        subtasks: list[SubTask] = []
        for i, cell in enumerate(cells, 1):
            cell_id = cell.id or f"cell_{i}"
            subtasks.append(
                {
                    "id": f"task_{i}{suffix}",
                    "question": cell.question,
                    "rationale": "; ".join(cell.success_criteria)
                    or f"Fill research cell {cell.entity} {cell.dimension}".strip(),
                    "dependencies": [],
                    "status": "pending",
                    "cell_id": cell_id,
                    "entity": cell.entity,
                    "dimension": cell.dimension,
                    "target_queries": cell.target_queries or [cell.question],
                    "required_evidence": cell.required_evidence,
                    "success_criteria": cell.success_criteria,
                    "allow_insufficient_data": cell.allow_insufficient_data,
                }
            )
    else:
        subtasks = [
            {
                "id": f"{t.id}{suffix}",
                "question": t.question,
                "rationale": t.rationale,
                "dependencies": [f"{d}{suffix}" if suffix else d for d in t.dependencies],
                "status": "pending",
                "cell_id": f"{t.id}{suffix}",
                "entity": "",
                "dimension": "general",
                "target_queries": [t.question],
                "required_evidence": 1,
                "success_criteria": [t.rationale],
                "allow_insufficient_data": False,
            }
            for t in plan.tasks[:MAX_SUBTASKS]
        ]

    research_plan = plan.model_dump()
    if research_plan.get("research_cells"):
        for i, cell in enumerate(research_plan["research_cells"][: len(subtasks)]):
            cell["task_id"] = subtasks[i]["id"]
    else:
        research_plan["research_cells"] = [
            {
                "id": t["cell_id"],
                "task_id": t["id"],
                "entity": t.get("entity", ""),
                "dimension": t.get("dimension", "general"),
                "question": t["question"],
                "target_queries": t.get("target_queries", [t["question"]]),
                "required_evidence": t.get("required_evidence", 1),
                "success_criteria": t.get("success_criteria", []),
                "evidence_type": "factual",
                "allow_insufficient_data": t.get("allow_insufficient_data", False),
            }
            for t in subtasks
        ]

    log_event(
        "planner",
        state.get("session_id", "-"),
        iteration=iteration + 1,
        subtasks=len(subtasks),
        development=False,
    )
    return {
        "plan": subtasks,
        "research_plan": research_plan,
        "current_iteration": iteration + 1,
        "gap_rounds": 0,
        "total_tokens_used": tokens,
    }
