"""Synthesizer agent: merges findings into a coherent markdown report.

Citation map is built deterministically from claims emitted by researchers,
not from URLs extracted out of the LLM's free-text output. Any [N] token the
LLM emits that does not refer to a known citation is dropped, and surviving
[N] tokens are renumbered so the report stays consistent with `## Sources`.
"""
from __future__ import annotations

import re

from app.config import get_synthesizer_llm, is_development
from app.observability import tokens_from_response
from app.prompts import SYNTHESIZER_PROMPT
from app.prompts.synthesizer import PROMPT_VERSION
from app.state import AgentState

_CITE_RE = re.compile(r"\[(\d+)\]")


def _build_citation_map(findings: list[dict]) -> dict[str, int]:
    """Assign 1-based numbers to every unique claim source URL across findings.

    Falls back to `Finding.sources` URLs when a finding produced no claims, so
    older runs (and dev-mode stubs) still cite something."""
    seen: dict[str, int] = {}
    next_id = 1
    for f in findings:
        for claim in (f.get("claims") or []):
            url = (claim.get("source_url") or "").rstrip(".,);:")
            if url and url not in seen:
                seen[url] = next_id
                next_id += 1
        if not f.get("claims"):
            for url in (f.get("sources") or []):
                url = url.rstrip(".,);:")
                if url and url not in seen:
                    seen[url] = next_id
                    next_id += 1
    return seen


def _format_findings_for_prompt(findings: list[dict], cmap: dict[str, int]) -> str:
    if not findings:
        return "(no findings — researchers produced no results)"
    chunks: list[str] = []
    for f in findings:
        header = (
            f"--- Finding {f.get('task_id', '?')} "
            f"(confidence={f.get('confidence', 0):.2f}) ---"
        )
        body = (f.get("content") or "").strip()
        claim_lines: list[str] = []
        for c in (f.get("claims") or []):
            url = (c.get("source_url") or "").rstrip(".,);:")
            n = cmap.get(url)
            if n is None:
                continue
            stmt = c.get("statement", "").strip()
            snip = (c.get("snippet") or "").strip()[:200]
            claim_lines.append(f"  - [{n}] {stmt}  (snippet: \"{snip}\")")
        claims_block = "\n".join(claim_lines) if claim_lines else "  (no structured claims)"
        chunks.append(f"{header}\nNarrative: {body}\nClaims:\n{claims_block}")
    return "\n\n".join(chunks)


def _renumber_and_filter(draft: str, cmap: dict[str, int]) -> tuple[str, dict[str, int]]:
    """Drop [N] tokens whose N is not in cmap; renumber survivors 1..K in order
    of first appearance; return updated draft and a fresh url→N map."""
    inverse = {n: url for url, n in cmap.items()}

    # Pass 1: drop orphan [N] tokens but keep claim numbers we know about.
    def _filter(match: re.Match) -> str:
        n = int(match.group(1))
        if n not in inverse:
            return ""  # orphan → strip
        return match.group(0)

    cleaned = _CITE_RE.sub(_filter, draft)

    # Pass 2: renumber in order of appearance for compactness.
    appearance: dict[int, int] = {}
    counter = [0]

    def _renumber(match: re.Match) -> str:
        old = int(match.group(1))
        if old not in appearance:
            counter[0] += 1
            appearance[old] = counter[0]
        return f"[{appearance[old]}]"

    renumbered = _CITE_RE.sub(_renumber, cleaned)

    new_cmap: dict[str, int] = {}
    for old, new in appearance.items():
        url = inverse.get(old)
        if url:
            new_cmap[url] = new
    return renumbered, new_cmap


def _append_sources_section(report: str, cmap: dict[str, int]) -> str:
    if not cmap:
        return report
    # If the LLM already wrote a `## Sources` section, replace it.
    sources_lines = [f"[{n}] {url}" for url, n in sorted(cmap.items(), key=lambda kv: kv[1])]
    block = "## Sources\n" + "\n".join(sources_lines)

    if re.search(r"(?im)^\#{1,3}\s+Sources\b", report):
        # Truncate at the first Sources heading, then append our deterministic block.
        report = re.split(r"(?im)^\#{1,3}\s+Sources\b.*$", report, maxsplit=1)[0].rstrip()
    return f"{report}\n\n{block}\n"


def synthesizer_node(state: AgentState) -> dict:
    findings = state.get("findings", [])

    if is_development():
        lines = [
            f"# Development Report: {state['user_query']}",
            "",
            "This report was generated in development mode without external LLM, search, or fetch calls.",
            "",
            "## Findings",
        ]
        if findings:
            for f in findings:
                lines.append(f"- {f.get('task_id', 'task')}: {f.get('content', '').strip()}")
        else:
            lines.append("- No findings were produced.")
        return {"draft_report": "\n".join(lines), "citations": []}

    cmap = _build_citation_map(findings)
    prompt = SYNTHESIZER_PROMPT.format(
        query=state["user_query"],
        findings=_format_findings_for_prompt(findings, cmap),
    )

    llm = get_synthesizer_llm()
    response = llm.invoke(
        prompt,
        config={"metadata": {"prompt_version": PROMPT_VERSION, "agent": "synthesizer"}},
    )
    draft = response.content if hasattr(response, "content") else str(response)
    tokens = tokens_from_response(response)

    draft, final_cmap = _renumber_and_filter(draft, cmap)
    draft = _append_sources_section(draft, final_cmap)

    citations = [url for url, _ in sorted(final_cmap.items(), key=lambda kv: kv[1])]

    return {
        "draft_report": draft,
        "citations": citations,
        "total_tokens_used": tokens,
    }
