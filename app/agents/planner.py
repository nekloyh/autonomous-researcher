"""Planner agent: decomposes user query into independent sub-tasks."""
from __future__ import annotations

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import MAX_SUBTASKS, get_planner_llm, is_development
from app.observability import tokens_from_response
from app.prompts import PLANNER_PROMPT
from app.prompts.planner import PROMPT_VERSION
from app.state import AgentState, SubTask


class SubTaskPlan(BaseModel):
    id: str = Field(description="Unique identifier like 'task_1'")
    question: str = Field(description="Specific, researchable question")
    rationale: str = Field(description="Why this task is needed to answer the main query")
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of tasks that must complete before this one (rare, usually empty)",
    )


class ResearchPlan(BaseModel):
    reasoning: str = Field(description="Brief explanation of the decomposition strategy")
    tasks: list[SubTaskPlan] = Field(description=f"List of max {MAX_SUBTASKS} sub-tasks")


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8), reraise=True)
def _invoke_planner(prompt: str) -> tuple[ResearchPlan, int]:
    llm = get_planner_llm().with_structured_output(ResearchPlan, include_raw=True)
    out = llm.invoke(
        prompt,
        config={"metadata": {"prompt_version": PROMPT_VERSION, "agent": "planner"}},
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
        return {
            "plan": [
                {
                    "id": "task_1",
                    "question": f"Development-mode summary for: {query}",
                    "rationale": "Deterministic local stub; no external LLM calls.",
                    "dependencies": [],
                    "status": "pending",
                }
            ],
            "current_iteration": iteration + 1,
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
    subtasks: list[SubTask] = [
        {
            "id": f"{t.id}{suffix}",
            "question": t.question,
            "rationale": t.rationale,
            "dependencies": [f"{d}{suffix}" if suffix else d for d in t.dependencies],
            "status": "pending",
        }
        for t in plan.tasks[:MAX_SUBTASKS]
    ]

    return {
        "plan": subtasks,
        "current_iteration": iteration + 1,
        "total_tokens_used": tokens,
    }
