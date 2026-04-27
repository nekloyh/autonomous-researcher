"""Critic agent: reviews draft report, decides replan vs finish.

Uses Gemini (different model family from generators) to reduce same-model bias.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import MAX_ITERATIONS, get_critic_llm
from app.prompts import CRITIC_PROMPT
from app.prompts.critic import PROMPT_VERSION
from app.state import AgentState, Critique


class CritiqueOutput(BaseModel):
    is_complete: bool = Field(description="Is the report comprehensive enough to finalize?")
    quality_score: float = Field(ge=0, le=1, description="Overall quality 0-1")
    completeness: float = Field(ge=0, le=1)
    evidence: float = Field(ge=0, le=1)
    depth: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    structure: float = Field(ge=0, le=1)
    missing_info: list[str] = Field(
        default_factory=list,
        description="Specific unanswered sub-questions. Empty if complete.",
    )
    factual_errors: list[str] = Field(
        default_factory=list,
        description="Claims that seem unsupported or contradictory.",
    )
    suggestions: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="", description="One paragraph rationale.")


def _iteration_threshold(iteration: int) -> float:
    if iteration <= 1:
        return 0.85
    if iteration == 2:
        return 0.75
    return 0.65


def critic_node(state: AgentState) -> dict:
    iteration = state.get("current_iteration", 1)
    prev = "\n".join(
        f"- iter {i + 1}: score={c.get('quality_score', 0):.2f}, complete={c.get('is_complete')}"
        for i, c in enumerate(state.get("critiques", []))
    ) or "(none)"

    prompt = CRITIC_PROMPT.format(
        query=state["user_query"],
        report=state.get("draft_report", "(empty draft)"),
        iteration=iteration,
        max_iter=MAX_ITERATIONS,
        previous_critiques=prev,
    )

    llm = get_critic_llm().with_structured_output(CritiqueOutput)
    result: CritiqueOutput = llm.invoke(
        prompt,
        config={"metadata": {"prompt_version": PROMPT_VERSION, "agent": "critic"}},
    )

    threshold = _iteration_threshold(iteration)
    is_complete = bool(result.is_complete) or result.quality_score >= threshold
    if iteration >= MAX_ITERATIONS:
        is_complete = True

    critique: Critique = {
        "is_complete": is_complete,
        "quality_score": float(result.quality_score),
        "missing_info": list(result.missing_info or []),
        "factual_errors": list(result.factual_errors or []),
        "suggestions": list(result.suggestions or []),
    }
    return {"critiques": [critique]}


def should_continue(state: AgentState) -> Literal["replan", "finish"]:
    if state.get("current_iteration", 0) >= MAX_ITERATIONS:
        return "finish"
    critiques = state.get("critiques", [])
    if not critiques:
        return "finish"
    last = critiques[-1]
    if last.get("is_complete"):
        return "finish"
    if not last.get("missing_info"):
        return "finish"
    return "replan"
