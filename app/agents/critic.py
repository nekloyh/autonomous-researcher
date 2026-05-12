"""Critic agent: reviews draft report, decides replan vs finish.

Uses Gemini (different model family from generators) to reduce same-model bias.
Receives a structured summary of findings and per-claim coverage so it can flag
report assertions that have no supporting claim.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import MAX_ITERATIONS, get_critic_llm, is_development
from app.observability import tokens_from_response
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
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Statements in the report that have no matching evidence claim.",
    )
    suggestions: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="", description="One paragraph rationale.")


def _iteration_threshold(iteration: int) -> float:
    if iteration <= 1:
        return 0.85
    if iteration == 2:
        return 0.75
    return 0.65


def _format_findings_summary(findings: list[dict]) -> str:
    if not findings:
        return "(no findings)"
    lines: list[str] = []
    for f in findings:
        claims = f.get("claims") or []
        tid = f.get("task_id", "?")
        conf = f.get("confidence", 0)
        lines.append(f"### {tid} (confidence={conf:.2f}, {len(claims)} claims)")
        if not claims:
            lines.append("  - (no structured claims)")
        for c in claims[:8]:
            stmt = (c.get("statement") or "").strip()
            url = c.get("source_url") or ""
            cc = c.get("confidence", 0)
            lines.append(f"  - [{cc:.2f}] {stmt}  ←  {url}")
    return "\n".join(lines)


def critic_node(state: AgentState) -> dict:
    iteration = state.get("current_iteration", 1)
    max_iter = state.get("max_iterations") or MAX_ITERATIONS

    if is_development():
        critique: Critique = {
            "is_complete": True,
            "quality_score": 1.0,
            "missing_info": [],
            "factual_errors": [],
            "suggestions": ["Development-mode critic stub; no external model call."],
        }
        return {"critiques": [critique]}

    prev = "\n".join(
        f"- iter {i + 1}: score={c.get('quality_score', 0):.2f}, complete={c.get('is_complete')}"
        for i, c in enumerate(state.get("critiques", []))
    ) or "(none)"

    findings_block = _format_findings_summary(state.get("findings", []))

    prompt = CRITIC_PROMPT.format(
        query=state["user_query"],
        report=state.get("draft_report", "(empty draft)"),
        iteration=iteration,
        max_iter=max_iter,
        previous_critiques=prev,
        findings_summary=findings_block,
    )

    llm = get_critic_llm().with_structured_output(CritiqueOutput, include_raw=True)
    out = llm.invoke(
        prompt,
        config={"metadata": {"prompt_version": PROMPT_VERSION, "agent": "critic"}},
    )
    if isinstance(out, dict):
        result = out.get("parsed")
        tokens = tokens_from_response(out.get("raw"))
        if result is None and out.get("parsing_error"):
            raise out["parsing_error"]
    else:
        result = out
        tokens = 0

    threshold = _iteration_threshold(iteration)
    is_complete = bool(result.is_complete) or result.quality_score >= threshold
    if result.unsupported_claims and iteration < max_iter:
        is_complete = False
    if iteration >= max_iter:
        is_complete = True

    critique: Critique = {
        "is_complete": is_complete,
        "quality_score": float(result.quality_score),
        "missing_info": list(result.missing_info or []),
        "factual_errors": list(result.factual_errors or []) + list(result.unsupported_claims or []),
        "suggestions": list(result.suggestions or []),
    }
    return {"critiques": [critique], "total_tokens_used": tokens}


def should_continue(state: AgentState) -> Literal["replan", "finish"]:
    max_iter = state.get("max_iterations") or MAX_ITERATIONS
    if state.get("current_iteration", 0) >= max_iter:
        return "finish"
    critiques = state.get("critiques", [])
    if not critiques:
        return "finish"
    last = critiques[-1]
    if last.get("is_complete"):
        return "finish"
    if not last.get("missing_info") and not last.get("factual_errors"):
        return "finish"
    return "replan"
