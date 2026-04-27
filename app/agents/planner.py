"""Planner agent: decomposes user query into independent sub-tasks."""
from __future__ import annotations

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import MAX_SUBTASKS, get_planner_llm
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
def _invoke_planner(prompt: str) -> ResearchPlan:
    llm = get_planner_llm().with_structured_output(ResearchPlan)
    return llm.invoke(
        prompt,
        config={"metadata": {"prompt_version": PROMPT_VERSION, "agent": "planner"}},
    )


def planner_node(state: AgentState) -> dict:
    """Planning node — runs at start and on each replan."""
    iteration = state.get("current_iteration", 0)

    previous_context = ""
    if state.get("critiques"):
        last = state["critiques"][-1]
        gaps = "\n".join(f"- {m}" for m in last.get("missing_info", []))
        if gaps:
            previous_context = (
                "The previous draft was incomplete. Focus the new plan on filling "
                "ONLY these gaps; do not redo what is already covered:\n" + gaps
            )

    prompt = PLANNER_PROMPT.format(
        query=state["user_query"],
        previous_context=previous_context or "(none — this is the first iteration)",
        known_context="(no prior context)",
        max_tasks=MAX_SUBTASKS,
    )

    plan = _invoke_planner(prompt)

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
    }
