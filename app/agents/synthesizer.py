"""Synthesizer agent: merges findings into a coherent markdown report."""
from __future__ import annotations

import re

from app.config import get_synthesizer_llm
from app.prompts import SYNTHESIZER_PROMPT
from app.prompts.synthesizer import PROMPT_VERSION
from app.state import AgentState

_URL_RE = re.compile(r"https?://[^\s\)\]\>]+")


def _format_findings(findings: list[dict]) -> str:
    if not findings:
        return "(no findings — researchers produced no results)"
    chunks = []
    for f in findings:
        sources = ", ".join(f.get("sources", [])) or "(none)"
        chunks.append(
            f"--- Finding {f['task_id']} (confidence={f.get('confidence', 0):.2f}) ---\n"
            f"{f.get('content', '').strip()}\n"
            f"Sources: {sources}"
        )
    return "\n\n".join(chunks)


def _whitelist_citations(draft: str, findings: list[dict]) -> list[str]:
    allowed = set()
    for f in findings:
        for u in f.get("sources", []) or []:
            allowed.add(u.rstrip(".,);:"))

    cited = [u.rstrip(".,);:") for u in _URL_RE.findall(draft)]
    seen = set()
    out: list[str] = []
    for u in cited:
        if u not in allowed:
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def synthesizer_node(state: AgentState) -> dict:
    findings = state.get("findings", [])
    prompt = SYNTHESIZER_PROMPT.format(
        query=state["user_query"],
        findings=_format_findings(findings),
    )

    llm = get_synthesizer_llm()
    response = llm.invoke(
        prompt,
        config={"metadata": {"prompt_version": PROMPT_VERSION, "agent": "synthesizer"}},
    )
    draft = response.content if hasattr(response, "content") else str(response)

    citations = _whitelist_citations(draft, findings)

    return {
        "draft_report": draft,
        "citations": citations,
    }
