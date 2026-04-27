"""Researcher agent: ReAct loop over 4 tools, executes one sub-task."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from langgraph.prebuilt import create_react_agent

from app.config import REACT_MAX_STEPS, get_researcher_llm
from app.prompts import RESEARCHER_PROMPT
from app.prompts.researcher import PROMPT_VERSION
from app.state import Finding, ResearcherState
from app.tools import ALL_TOOLS

_URL_RE = re.compile(r"https?://[^\s\)\]]+")


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


def _confidence(num_sources: int, num_unique_domains: int) -> float:
    if num_sources == 0:
        return 0.3
    if num_unique_domains >= 3:
        return 0.9
    if num_unique_domains == 2:
        return 0.75
    return 0.6


def researcher_node(state: ResearcherState) -> dict:
    """Run the ReAct agent on one sub-task. Called via Send() in parallel."""
    task = state["task"]

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

    tool_calls = sum(
        1 for m in messages if getattr(m, "tool_calls", None)
    )

    finding: Finding = {
        "task_id": task["id"],
        "content": answer_text,
        "sources": urls,
        "confidence": _confidence(len(urls), len(domains)),
        "tool_calls": tool_calls,
    }

    return {
        "findings": [finding],
        "total_tool_calls": tool_calls,
    }
