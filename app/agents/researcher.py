"""Researcher agent: ReAct loop over 4 tools, executes one sub-task.

Two-stage pattern:
  1. ReAct loop with tool calls → narrative answer (existing behavior).
  2. Post-hoc structured-output pass that distils the conversation into a list
     of atomic Claim objects (statement, source_url, snippet, confidence).
Stage 2 lets us build deterministic citations downstream and lets the critic
spot unsupported claims.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from app.config import REACT_MAX_STEPS, get_researcher_llm, is_development
from app.observability import tokens_from_messages, tokens_from_response
from app.prompts import RESEARCHER_PROMPT
from app.prompts.researcher import PROMPT_VERSION
from app.state import Claim, Finding, ResearcherState
from app.tools import ALL_TOOLS

_URL_RE = re.compile(r"https?://[^\s\)\]]+")


class ClaimModel(BaseModel):
    statement: str = Field(description="One atomic factual claim, ≤30 words.")
    source_url: str = Field(description="The URL that supports this claim (must be one you actually fetched).")
    snippet: str = Field(description="≤300 chars verbatim from the source backing the claim.")
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class ResearcherStructuredOutput(BaseModel):
    narrative: str = Field(description="2-4 sentence synthesised answer to the sub-task.")
    claims: list[ClaimModel] = Field(
        default_factory=list,
        description="3-8 atomic claims, each tied to a URL you fetched. Drop a claim if you cannot identify a supporting URL.",
    )


def _build_react_agent():
    return create_react_agent(model=get_researcher_llm(), tools=ALL_TOOLS)


_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = _build_react_agent()
    return _agent


def _extract_urls(messages) -> list[str]:
    """Pull URLs from tool outputs (web_search results, fetch_url args)."""
    urls: list[str] = []
    for m in messages:
        if getattr(m, "type", None) == "tool":
            urls.extend(_URL_RE.findall(str(m.content or "")))
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls or []:
                args = tc.get("args") or {}
                if "url" in args and isinstance(args["url"], str):
                    urls.append(args["url"])
    seen = set()
    deduped = []
    for u in urls:
        u = u.rstrip(".,);:")
        if u in seen:
            continue
        seen.add(u)
        deduped.append(u)
    return deduped


def _messages_excerpt(messages, max_chars: int = 6000) -> str:
    """Compact ReAct transcript for the structured-output pass."""
    parts: list[str] = []
    for m in messages:
        mtype = getattr(m, "type", None)
        content = str(getattr(m, "content", "") or "")[:1500]
        if mtype == "tool":
            parts.append(f"[TOOL OUTPUT]\n{content}")
        elif mtype in ("ai", "assistant"):
            parts.append(f"[ASSISTANT]\n{content}")
    blob = "\n\n---\n\n".join(parts)
    if len(blob) > max_chars:
        blob = blob[-max_chars:]  # tail: most relevant evidence is near the end
    return blob


def _confidence(num_sources: int, num_unique_domains: int) -> float:
    if num_sources == 0:
        return 0.3
    if num_unique_domains >= 3:
        return 0.9
    if num_unique_domains == 2:
        return 0.75
    return 0.6


def _extract_claims(
    messages, allowed_urls: list[str], task_question: str
) -> tuple[list[Claim], str, int]:
    """Run a structured-output pass over the ReAct transcript.

    Returns (claims, narrative, tokens_used).
    """
    if not messages:
        return [], "", 0

    allowed = {u.rstrip(".,);:") for u in allowed_urls}
    transcript = _messages_excerpt(messages)
    allowed_block = "\n".join(f"- {u}" for u in sorted(allowed)) or "(none — no URLs were fetched)"

    extractor_prompt = (
        "You are an evidence extractor. Read the research transcript below and "
        "output a small list of atomic factual claims, each tied to ONE URL "
        "that actually appears in the allowed URL list.\n\n"
        f"## Sub-task being researched\n{task_question}\n\n"
        f"## Allowed URLs (you must pick `source_url` from this list)\n{allowed_block}\n\n"
        f"## Research transcript\n{transcript}\n\n"
        "Rules:\n"
        "- Each claim is ONE concrete fact (≤30 words), not opinion.\n"
        "- `source_url` MUST be exactly one URL from the allowed list above. "
        "If you cannot tie a claim to such a URL, drop it.\n"
        "- `snippet` MUST be ≤300 chars copied verbatim from the source.\n"
        "- Return 3-8 claims, or fewer if evidence is thin. Empty list is fine.\n"
        "- Also write a 2-4 sentence `narrative` answering the sub-task."
    )

    tokens = 0
    try:
        llm = get_researcher_llm().with_structured_output(
            ResearcherStructuredOutput, include_raw=True
        )
        out = llm.invoke(
            extractor_prompt,
            config={"metadata": {"prompt_version": PROMPT_VERSION, "agent": "researcher_extract"}},
        )
        if isinstance(out, dict):
            tokens = tokens_from_response(out.get("raw"))
            if out.get("parsing_error") and not out.get("parsed"):
                return [], "", tokens
            result = out.get("parsed")
        else:
            result = out
        if result is None:
            return [], "", tokens
    except Exception:
        return [], "", tokens

    claims: list[Claim] = []
    for c in result.claims or []:
        url = (c.source_url or "").rstrip(".,);:")
        if url not in allowed:
            continue
        claims.append(
            Claim(
                statement=c.statement.strip(),
                source_url=url,
                snippet=(c.snippet or "").strip()[:300],
                confidence=float(c.confidence),
            )
        )
    return claims, (result.narrative or "").strip(), tokens


def researcher_node(state: ResearcherState) -> dict:
    """Run the ReAct agent on one sub-task. Called via Send() in parallel."""
    task = state["task"]

    if is_development():
        finding: Finding = {
            "task_id": task["id"],
            "content": (
                f"Development-mode finding for '{task['question']}'. "
                "External LLM and tool calls are disabled."
            ),
            "claims": [],
            "sources": [],
            "confidence": 0.5,
            "tool_calls": 0,
        }
        return {"findings": [finding], "total_tool_calls": 0}

    system_msg = RESEARCHER_PROMPT.format(
        user_query=state["user_query"],
        question=task["question"],
        rationale=task["rationale"],
        max_steps=REACT_MAX_STEPS,
    )

    agent = _get_agent()
    try:
        result = agent.invoke(
            {
                "messages": [
                    ("system", system_msg),
                    ("user", f"Begin research on: {task['question']}"),
                ]
            },
            config={
                "recursion_limit": REACT_MAX_STEPS * 2 + 4,
                "metadata": {
                    "prompt_version": PROMPT_VERSION,
                    "agent": "researcher",
                    "task_id": task["id"],
                },
            },
        )
    except Exception as e:
        finding: Finding = {
            "task_id": task["id"],
            "content": f"Research failed: {type(e).__name__}: {e}",
            "claims": [],
            "sources": [],
            "confidence": 0.0,
            "tool_calls": 0,
        }
        return {"findings": [finding], "errors": [f"researcher[{task['id']}]: {e}"]}

    messages = result.get("messages", [])
    final_msg = messages[-1] if messages else None
    answer_text = getattr(final_msg, "content", "") or ""

    urls = _extract_urls(messages)
    domains = {urlparse(u).netloc for u in urls if urlparse(u).netloc}
    tool_calls = sum(1 for m in messages if getattr(m, "tool_calls", None))
    react_tokens = tokens_from_messages(messages)

    claims, narrative, extract_tokens = _extract_claims(messages, urls, task["question"])
    content = narrative or answer_text

    finding: Finding = {
        "task_id": task["id"],
        "content": content,
        "claims": claims,
        "sources": urls,
        "confidence": _confidence(len(urls), len(domains)),
        "tool_calls": tool_calls,
    }

    return {
        "findings": [finding],
        "total_tool_calls": tool_calls,
        "total_tokens_used": react_tokens + extract_tokens,
    }
