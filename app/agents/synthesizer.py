"""Synthesizer agent: merges findings into a coherent markdown report.

Citation map is built deterministically from claims emitted by researchers,
not from URLs extracted out of the LLM's free-text output. Any [N] token the
LLM emits that does not refer to a known citation is dropped, and surviving
[N] tokens are renumbered so the report stays consistent with `## Sources`.
"""
from __future__ import annotations

import re

from app.config import get_synthesizer_llm, is_development
from app.entity_guard import entities_in_text, is_comparison_query, is_entity_contaminated
from app.logger import log_event
from app.observability import tokens_from_response
from app.prompts import SYNTHESIZER_PROMPT
from app.prompts.synthesizer import PROMPT_VERSION
from app.provider_rotation import invoke_with_rotation, rotate_groq
from app.research_plan import build_cell_coverage
from app.source_policy import classify_policy_tier
from app.state import AgentState

_CITE_RE = re.compile(r"\[(\d+)\]")
_COVERAGE_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "products/services": ("product", "service", "solution", "launch", "announce"),
    "partnerships": ("partner", "partnership", "collaboration", "alliance"),
    "infrastructure/investment": ("infrastructure", "investment", "cloud", "data center", "gpu"),
    "market/customers": ("market", "customer", "user", "mau", "adoption"),
    "financial impact": ("revenue", "profit", "loss", "financial", "growth"),
}


def _build_citation_map(findings: list[dict]) -> dict[str, int]:
    """Assign 1-based numbers to every unique claim source URL across findings.

    Only structured, sourced claims are eligible. Narrative-only findings and
    raw `Finding.sources` URLs are not enough to become citations."""
    seen: dict[str, int] = {}
    next_id = 1
    for f in findings:
        for claim in (f.get("claims") or []):
            url = (claim.get("source_url") or "").rstrip(".,);:")
            stmt = (claim.get("statement") or "").strip()
            if url and stmt and url not in seen:
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
        sub_q = (f.get("sub_question") or "").strip()
        if sub_q:
            header += f"\nSub-question: {sub_q}"
        gaps = [str(g).strip() for g in (f.get("gaps") or []) if str(g).strip()]
        claim_lines: list[str] = []
        for c in (f.get("claims") or []):
            url = (c.get("source_url") or "").rstrip(".,);:")
            n = cmap.get(url)
            if n is None:
                continue
            stmt = c.get("statement", "").strip()
            snip = (c.get("snippet") or "").strip()[:200]
            meta = []
            if c.get("cell_id"):
                meta.append(f"cell={c.get('cell_id')}")
            if c.get("entity"):
                meta.append(f"entity={c.get('entity')}")
            if c.get("dimension"):
                meta.append(f"dimension={c.get('dimension')}")
            meta_text = f" ({'; '.join(meta)})" if meta else ""
            claim_lines.append(f"  - [{n}] {stmt}{meta_text}  (snippet: \"{snip}\")")
        claims_block = "\n".join(claim_lines) if claim_lines else "  (no structured claims)"
        gaps_block = "\n".join(f"  - {g}" for g in gaps) if gaps else "  (none)"
        chunks.append(f"{header}\nClaims:\n{claims_block}\nKnown gaps:\n{gaps_block}")
    return "\n\n".join(chunks)


def _filter_findings_for_synthesis(findings: list[dict], query: str) -> tuple[list[dict], dict[str, int]]:
    stats = {
        "claims_in": 0,
        "claims_kept": 0,
        "dropped_unsourced": 0,
        "dropped_entity_contamination": 0,
        "dropped_duplicate_claims": 0,
        "dropped_blocked_source": 0,
        "dropped_invalid_validation": 0,
    }
    filtered: list[dict] = []
    seen_claims: set[tuple[str, str]] = set()
    for finding in findings or []:
        kept_claims: list[dict] = []
        for claim in finding.get("claims") or []:
            stats["claims_in"] += 1
            url = (claim.get("source_url") or "").rstrip(".,);:")
            stmt = (claim.get("statement") or "").strip()
            snippet = (claim.get("snippet") or "").strip()
            if not url or not stmt:
                stats["dropped_unsourced"] += 1
                continue
            if claim.get("validation_status", "valid") == "dropped":
                stats["dropped_invalid_validation"] += 1
                continue
            if claim.get("source_policy_tier") == "blocked" or classify_policy_tier(
                url,
                snippet=snippet,
                query=query,
            ) == "blocked":
                stats["dropped_blocked_source"] += 1
                continue
            if is_entity_contaminated(query, f"{stmt}\n{snippet}", url):
                stats["dropped_entity_contamination"] += 1
                continue
            key = (re.sub(r"\s+", " ", stmt.lower()), url)
            if key in seen_claims:
                stats["dropped_duplicate_claims"] += 1
                continue
            seen_claims.add(key)
            kept_claims.append({**claim, "source_url": url, "statement": stmt, "snippet": snippet})
        if kept_claims:
            filtered.append({**finding, "claims": kept_claims})
            stats["claims_kept"] += len(kept_claims)
    return filtered, stats


def _claim_dimension(text: str) -> str:
    lower = (text or "").lower()
    for dimension, terms in _COVERAGE_DIMENSIONS.items():
        if any(term in lower for term in terms):
            return dimension
    return "general"


def _format_coverage_matrix(findings: list[dict], query: str) -> str:
    if not is_comparison_query(query):
        return ""
    entities = sorted(entities_in_text(query))
    if len(entities) < 2:
        return ""
    coverage = {
        entity: {dimension: 0 for dimension in _COVERAGE_DIMENSIONS}
        for entity in entities
    }
    for finding in findings:
        for claim in finding.get("claims") or []:
            text = f"{claim.get('statement', '')} {claim.get('snippet', '')} {claim.get('source_url', '')}"
            mentioned = set(claim.get("attributed_entities") or []) | entities_in_text(text)
            dimension = _claim_dimension(text)
            if dimension not in _COVERAGE_DIMENSIONS:
                continue
            for entity in mentioned & set(entities):
                coverage[entity][dimension] += 1
    lines = ["\n\n--- Comparison coverage matrix ---"]
    lines.append(
        "For any table cell with 0 validated claims, write exactly `Insufficient data found`."
    )
    lines.append("| Entity | Dimension | Validated claims |")
    lines.append("| --- | --- | --- |")
    for entity in entities:
        for dimension, count in coverage[entity].items():
            lines.append(f"| {entity} | {dimension} | {count} |")
    return "\n".join(lines)


def _format_plan_coverage(research_plan: dict, findings: list[dict]) -> str:
    if not research_plan or not research_plan.get("research_cells"):
        return ""
    coverage = build_cell_coverage(research_plan, findings)
    lines = ["\n\n--- Research plan coverage ---"]
    if research_plan.get("query_intent"):
        lines.append(f"Intent: {research_plan.get('query_intent')}")
    if research_plan.get("synthesis_requirements"):
        lines.append("Synthesis requirements:")
        lines.extend(f"- {item}" for item in research_plan.get("synthesis_requirements") or [])
    lines.append(
        "For any required cell that is not filled, write exactly "
        "`Insufficient verified data after targeted research` and explain the missing evidence."
    )
    lines.append("| Cell | Entity | Dimension | Required evidence | Validated claims | Status |")
    lines.append("| --- | --- | --- | ---: | ---: | --- |")
    for cell in coverage:
        lines.append(
            "| {cell_id} | {entity} | {dimension} | {required} | {count} | {status} |".format(
                cell_id=cell.get("cell_id") or cell.get("id"),
                entity=cell.get("entity") or "-",
                dimension=cell.get("dimension") or "-",
                required=cell.get("required_evidence") or 1,
                count=cell.get("validated_claims") or 0,
                status=cell.get("cell_status") or "unfilled",
            )
        )
    return "\n".join(lines)


def _empty_evidence_report(query: str) -> str:
    return (
        f"# Report: {query}\n\n"
        "## Executive Summary\n"
        "not found in available sources.\n\n"
        "## Findings\n"
        "not found in available sources.\n\n"
        "## Limitations / Unknowns\n"
        "No source-backed claims were available, so the system cannot provide a verified answer.\n"
    )


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
        log_event(
            "synthesizer",
            state.get("session_id", "-"),
            claims_in=sum(len(f.get("claims") or []) for f in findings),
            citations_out=0,
            dropped_citations=0,
            development=True,
        )
        return {"draft_report": "\n".join(lines), "citations": []}

    sourced_findings, stats = _filter_findings_for_synthesis(findings, state["user_query"])
    cmap = _build_citation_map(sourced_findings)
    if not cmap:
        draft = _empty_evidence_report(state["user_query"])
        log_event(
            "synthesizer",
            state.get("session_id", "-"),
            claims_in=stats["claims_in"],
            claims_kept=stats["claims_kept"],
            citations_out=0,
            dropped_citations=0,
            dropped_unsourced=stats["dropped_unsourced"],
            dropped_entity=stats["dropped_entity_contamination"],
            dropped_duplicate=stats["dropped_duplicate_claims"],
            no_sourced_claims=True,
        )
        return {"draft_report": draft, "citations": []}

    prompt = SYNTHESIZER_PROMPT.format(
        query=state["user_query"],
        findings=(
            _format_findings_for_prompt(sourced_findings, cmap)
            + (
                _format_plan_coverage(state.get("research_plan", {}), sourced_findings)
                or _format_coverage_matrix(sourced_findings, state["user_query"])
            )
        ),
    )

    response = invoke_with_rotation(
        "groq",
        lambda: get_synthesizer_llm().invoke(
            prompt,
            config={"metadata": {"prompt_version": PROMPT_VERSION, "agent": "synthesizer"}},
        ),
        attempts=2,
        rotate=rotate_groq,
    )
    draft = response.content if hasattr(response, "content") else str(response)
    tokens = tokens_from_response(response)

    valid_numbers = set(cmap.values())
    dropped_citations = sum(
        1 for n in _CITE_RE.findall(draft) if int(n) not in valid_numbers
    )
    draft, final_cmap = _renumber_and_filter(draft, cmap)
    draft = _append_sources_section(draft, final_cmap)

    citations = [url for url, _ in sorted(final_cmap.items(), key=lambda kv: kv[1])]

    log_event(
        "synthesizer",
        state.get("session_id", "-"),
        claims_in=stats["claims_in"],
        claims_kept=stats["claims_kept"],
        citations_out=len(citations),
        dropped_citations=dropped_citations,
        dropped_unsourced=stats["dropped_unsourced"],
        dropped_entity=stats["dropped_entity_contamination"],
        dropped_duplicate=stats["dropped_duplicate_claims"],
        dropped_blocked=stats["dropped_blocked_source"],
        dropped_invalid=stats["dropped_invalid_validation"],
        development=False,
    )
    return {
        "draft_report": draft,
        "citations": citations,
        "total_tokens_used": tokens,
    }
