"""Critic agent: reviews draft report, decides replan vs finish.

Uses Gemini (different model family from generators) to reduce same-model bias.
Receives a structured summary of findings and per-claim coverage so it can flag
report assertions that have no supporting claim.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import MAX_ITERATIONS, get_critic_llm, is_development
from app.logger import log_event
from app.observability import tokens_from_response
from app.prompts import CRITIC_PROMPT
from app.prompts.critic import PROMPT_VERSION
from app.provider_rotation import invoke_with_rotation, rotate_google
from app.state import AgentState, Critique


class GapOutput(BaseModel):
    question: str = Field(description="Specific follow-up question to research.")
    origin_task_id: str = Field(default="", description="Task id related to the gap, if known.")
    reason: str = Field(default="", description="Why this gap blocks a verified answer.")
    priority: Literal["high", "medium", "low"] = "medium"


class CritiqueOutput(BaseModel):
    action: Literal["finalize", "research_gaps", "replan"] = Field(
        description="Route decision for the graph."
    )
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
    gaps: list[GapOutput] = Field(
        default_factory=list,
        description="Structured targeted follow-up questions when action=research_gaps.",
    )
    factual_errors: list[str] = Field(
        default_factory=list,
        description="Claims that seem unsupported or contradictory.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Statements in the report that have no matching evidence claim.",
    )
    conflicting_claims: list[str] = Field(
        default_factory=list,
        description="Evidence-backed claims that appear to contradict each other.",
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
        sub_q = (f.get("sub_question") or f.get("question") or "").strip()
        conf = f.get("confidence", 0)
        heading = f"### {tid} (confidence={conf:.2f}, {len(claims)} claims)"
        if sub_q:
            heading += f"\nSub-question: {sub_q}"
        lines.append(heading)
        if not claims:
            lines.append("  - (no structured claims)")
        for c in claims[:8]:
            stmt = (c.get("statement") or "").strip()
            url = c.get("source_url") or ""
            cc = c.get("confidence", 0)
            lines.append(f"  - [{cc:.2f}] {stmt}  ←  {url}")
    return "\n".join(lines)


def _critic_failure_update(state: AgentState, exc: Exception, tokens: int = 0) -> dict:
    iteration = state.get("current_iteration", 1)
    max_iter = state.get("max_iterations") or MAX_ITERATIONS
    hard_stop = iteration >= max_iter
    err = f"critic/provider failure: {type(exc).__name__}: {exc}"
    critique: Critique = {
        "action": "finalize" if hard_stop else "replan",
        "is_complete": hard_stop,
        "quality_score": 0.0,
        "missing_info": []
        if hard_stop
        else ["Critic provider failed; re-run critique after additional research."],
        "factual_errors": [err],
        "unsupported_claims": [],
        "conflicting_claims": [],
        "gaps": []
        if hard_stop
        else [
            {
                "question": "Can the draft be reviewed by the critic provider successfully?",
                "origin_task_id": "critic",
                "reason": err,
                "priority": "high",
            }
        ],
        "suggestions": [
            "Hard stop reached; finalize only as unverified."
            if hard_stop
            else "Do not finalize this draft until critic review succeeds."
        ],
    }
    log_event(
        "critic",
        state.get("session_id", "-"),
        score=0.0,
        complete=hard_stop,
        failure_reason=err,
        hard_stop=hard_stop,
    )
    update = {"critiques": [critique], "errors": [err]}
    if tokens:
        update["total_tokens_used"] = tokens
    return update


def critic_node(state: AgentState) -> dict:
    iteration = state.get("current_iteration", 1)
    max_iter = state.get("max_iterations") or MAX_ITERATIONS

    if is_development():
        critique: Critique = {
            "action": "finalize",
            "is_complete": True,
            "quality_score": 1.0,
            "missing_info": [],
            "gaps": [],
            "factual_errors": [],
            "unsupported_claims": [],
            "conflicting_claims": [],
            "suggestions": ["Development-mode critic stub; no external model call."],
        }
        log_event(
            "critic",
            state.get("session_id", "-"),
            score=critique["quality_score"],
            complete=critique["is_complete"],
            failure_reason="",
        )
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

    try:
        def _call():
            llm = get_critic_llm().with_structured_output(CritiqueOutput, include_raw=True)
            return llm.invoke(
                prompt,
                config={"metadata": {"prompt_version": PROMPT_VERSION, "agent": "critic"}},
            )

        out = invoke_with_rotation(
            "google",
            _call,
            attempts=2,
            rotate=rotate_google,
        )
        if isinstance(out, dict):
            result = out.get("parsed")
            tokens = tokens_from_response(out.get("raw"))
            if result is None and out.get("parsing_error"):
                raise out["parsing_error"]
        else:
            result = out
            tokens = 0
        if result is None:
            raise ValueError("critic returned no parsed result")
    except Exception as e:
        return _critic_failure_update(state, e)

    threshold = _iteration_threshold(iteration)
    score = float(result.quality_score)
    gap_questions = [g.question for g in result.gaps or [] if g.question.strip()]
    missing_questions = [m for m in result.missing_info or [] if str(m).strip()]
    has_targeted_gaps = bool(gap_questions or missing_questions or result.unsupported_claims)
    if iteration >= max_iter:
        action: Literal["finalize", "research_gaps", "replan"] = "finalize"
    elif score >= threshold and not result.unsupported_claims and not result.conflicting_claims:
        action = "finalize"
    elif score >= 0.5 and has_targeted_gaps:
        action = "research_gaps"
    else:
        action = "replan"

    # Honor stricter model decisions, but do not let it finalize below threshold.
    if iteration < max_iter and result.action == "replan":
        action = "replan"
    elif iteration < max_iter and result.action == "research_gaps" and has_targeted_gaps and score >= 0.45:
        action = "research_gaps"

    is_complete = action == "finalize"
    gaps = [
        {
            "question": g.question,
            "origin_task_id": g.origin_task_id,
            "reason": g.reason,
            "priority": g.priority,
        }
        for g in result.gaps or []
        if g.question.strip()
    ]
    for question in missing_questions:
        if question not in {g["question"] for g in gaps}:
            gaps.append(
                {
                    "question": question,
                    "origin_task_id": "",
                    "reason": "Critic marked this information as missing.",
                    "priority": "high",
                }
            )
    for claim in result.unsupported_claims or []:
        question = f"Find source-backed evidence for or against this report claim: {claim}"
        if question not in {g["question"] for g in gaps}:
            gaps.append(
                {
                    "question": question,
                    "origin_task_id": "",
                    "reason": "Unsupported report claim.",
                    "priority": "high",
                }
            )

    critique: Critique = {
        "action": action,
        "is_complete": is_complete,
        "quality_score": score,
        "missing_info": list(result.missing_info or []),
        "gaps": gaps,
        "factual_errors": list(result.factual_errors or []) + list(result.unsupported_claims or []),
        "unsupported_claims": list(result.unsupported_claims or []),
        "conflicting_claims": list(result.conflicting_claims or []),
        "suggestions": list(result.suggestions or []),
    }
    log_event(
        "critic",
        state.get("session_id", "-"),
        score=score,
        complete=is_complete,
        action=action,
        missing_info=len(critique["missing_info"]),
        factual_errors=len(critique["factual_errors"]),
        failure_reason="",
    )
    return {"critiques": [critique], "total_tokens_used": tokens}


def should_continue(state: AgentState) -> Literal["replan", "finish"]:
    max_iter = state.get("max_iterations") or MAX_ITERATIONS
    if state.get("current_iteration", 0) >= max_iter:
        return "finish"
    critiques = state.get("critiques", [])
    if not critiques:
        return "finish"
    last = critiques[-1]
    action = last.get("action")
    if action == "replan" or action == "research_gaps":
        return "replan"
    if action == "finalize":
        return "finish"
    if last.get("is_complete"):
        return "finish"
    return "replan"
