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
from collections import Counter
from types import SimpleNamespace
from urllib.parse import urlparse

from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from app.claim_validation import validate_claims
from app.config import (
    MAX_DIRECT_FETCHES_PER_TASK,
    REACT_MAX_STEPS,
    get_extractor_llm,
    get_researcher_llm,
    is_development,
)
from app.entity_guard import is_entity_contaminated
from app.logger import log_event
from app.long_document import is_long_document_url, relevant_document_excerpt
from app.observability import tokens_from_messages, tokens_from_response
from app.prompts import RESEARCHER_PROMPT
from app.prompts.researcher import PROMPT_VERSION
from app.provider_rotation import invoke_with_rotation, rotate_groq
from app.source_policy import classify_policy_tier
from app.state import Claim, Finding, ResearcherState
from app.tools import ALL_TOOLS
from app.tools.fetch_url import fetch_url

_URL_RE = re.compile(r"https?://[^\s\)\]]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class ClaimModel(BaseModel):
    statement: str = Field(description="One atomic factual claim, ≤30 words.")
    source_url: str = Field(description="The URL that supports this claim (must be one you actually fetched).")
    snippet: str = Field(description="≤300 chars verbatim from the source backing the claim.")
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class ResearcherStructuredOutput(BaseModel):
    narrative: str = Field(description="2-4 sentence synthesised answer to the sub-task.")
    claims: list[ClaimModel] = Field(
        default_factory=list,
        description="5-12 atomic claims, each tied to a URL you fetched. Drop a claim if you cannot identify a supporting URL.",
    )


def _build_react_agent():
    return create_react_agent(model=get_researcher_llm(), tools=ALL_TOOLS)


_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = _build_react_agent()
    return _agent


def _reset_agent() -> None:
    global _agent
    _agent = None


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


def _messages_excerpt(messages, max_chars: int = 12000) -> str:
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


def _format_assigned_sources(sources: list[dict]) -> str:
    if not sources:
        return "(none assigned; use web_search to discover sources)"
    lines = []
    for i, source in enumerate(sources[:6], 1):
        title = source.get("title") or "(untitled)"
        url = source.get("url") or source.get("canonical_url") or ""
        score = float(source.get("rank_score", 0.0))
        snippet = (source.get("snippet") or "").replace("\n", " ")[:220]
        lines.append(f"{i}. {title} — {url} (score={score:.1f})\n   {snippet}")
    return "\n".join(lines)


def _source_quality(sources: list[dict]) -> float:
    if not sources:
        return 0.0
    best = max(float(s.get("rank_score", 0.0)) for s in sources)
    return max(0.0, min(1.0, best / 12.0))


def _direct_fetch_assigned_sources(sources: list[dict]) -> tuple[list, list[str], int]:
    return _direct_fetch_assigned_sources_for_task(sources, {})


def _direct_fetch_assigned_sources_for_task(
    sources: list[dict], task: dict
) -> tuple[list, list[str], int]:
    messages = []
    urls: list[str] = []
    target_terms = [
        task.get("question", ""),
        task.get("entity", ""),
        task.get("dimension", ""),
        " ".join(task.get("target_queries", []) or []),
    ]
    for source in sources[:MAX_DIRECT_FETCHES_PER_TASK]:
        url = (source.get("url") or source.get("canonical_url") or "").strip()
        if not url:
            continue
        if is_long_document_url(url):
            content = relevant_document_excerpt(url, target_terms)
        else:
            content = fetch_url.invoke({"url": url})
        messages.append(
            SimpleNamespace(
                type="tool",
                content=(
                    f"URL: {url}\n"
                    f"TITLE: {source.get('title', '')}\n"
                    f"SNIPPET: {source.get('snippet', '')}\n\n"
                    f"{content}"
                ),
            )
        )
        if not str(content).startswith("ERROR:"):
            urls.append(url.rstrip(".,);:"))
    if messages:
        messages.append(
            SimpleNamespace(
                type="assistant",
                content="Collected assigned SourceBroker pages for evidence extraction.",
            )
        )
    return messages, list(dict.fromkeys(urls)), len(messages)


def _trim_words(text: str, max_words: int = 30) -> str:
    words = re.sub(r"\s+", " ", text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(".,;:") + "."


def _fallback_claims_from_sources(
    sources: list[dict], allowed_urls: list[str], user_query: str
) -> list[Claim]:
    """Create conservative source-backed claims from broker snippets.

    This is a degradation path for quota-constrained small models that sometimes
    fail structured extraction even when SourceBroker found good pages.
    """
    allowed = {u.rstrip(".,);:") for u in allowed_urls}
    claims: list[Claim] = []
    seen: set[tuple[str, str]] = set()
    for source in sources[:MAX_DIRECT_FETCHES_PER_TASK]:
        url = (source.get("url") or source.get("canonical_url") or "").rstrip(".,);:")
        if (
            url not in allowed
            or source.get("source_type") == "generic"
            or source.get("source_policy_tier") == "blocked"
            or classify_policy_tier(
                url,
                source.get("title", ""),
                source.get("snippet", ""),
                user_query,
            )
            == "blocked"
        ):
            continue
        title = re.sub(r"\s+", " ", source.get("title") or "").strip()
        snippet = re.sub(r"\s+", " ", source.get("snippet") or "").strip()
        support = snippet or title
        if not support:
            continue
        text = f"{title}. {snippet}" if title and snippet else support
        sentences = [
            s.strip(" -")
            for s in _SENTENCE_SPLIT_RE.split(text)
            if len(s.strip().split()) >= 5
        ]
        statement = _trim_words(sentences[0] if sentences else text)
        if not statement:
            continue
        if is_entity_contaminated(user_query, f"{statement}\n{support}", url):
            continue
        key = (statement.lower(), url)
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            Claim(
                statement=statement,
                source_url=url,
                snippet=support[:300],
                confidence=0.55,
            )
        )
    return claims


def _validation_gap_texts(gaps: list[dict]) -> list[str]:
    return [str(g.get("question", "")).strip() for g in gaps if str(g.get("question", "")).strip()]


def _direct_fetch_finding(
    task: dict,
    assigned_sources: list[dict],
    state: ResearcherState,
    *,
    recovered_from_error: bool = False,
    failure_reason: str = "",
) -> dict:
    messages, urls, direct_tool_calls = _direct_fetch_assigned_sources_for_task(
        assigned_sources, task
    )
    domains = {urlparse(u).netloc for u in urls if urlparse(u).netloc}
    claims, narrative, extract_tokens, extract_stats = _extract_claims(
        messages, urls, task["question"], state["user_query"]
    )
    fallback_claims = 0
    if not claims:
        claims = _fallback_claims_from_sources(assigned_sources, urls, state["user_query"])
        fallback_claims = len(claims)
    validation = validate_claims(
        claims,
        user_query=state["user_query"],
        task_question=task["question"],
        source_candidates=assigned_sources,
        cell_id=task.get("cell_id", task["id"]),
        entity=task.get("entity", ""),
        dimension=task.get("dimension", ""),
    )
    valid_claims = validation["valid_claims"]
    validation_gaps = validation["gaps"]
    content = narrative or (
        f"Fetched {len(urls)} assigned sources for '{task['question']}'. "
        + (
            f"Extracted {fallback_claims} conservative source-backed claims from snippets."
            if fallback_claims
            else "No source-backed claims were extracted."
        )
    )
    gaps = _validation_gap_texts(validation_gaps)
    if not valid_claims:
        gaps.append(f"No source-backed claims extracted for: {task['question']}")
    researcher_error_status = "none"
    required_evidence = int(task.get("required_evidence", 1) or 1)
    if recovered_from_error:
        researcher_error_status = "recovered" if len(valid_claims) >= required_evidence else "unrecovered"
    elif len(valid_claims) < required_evidence:
        researcher_error_status = "unrecovered"
    finding: Finding = {
        "task_id": task["id"],
        "sub_question": task["question"],
        "answer": content,
        "content": content,
        "claims": valid_claims,
        "sources": urls,
        "gaps": gaps,
        "confidence": _confidence(len(urls), len(domains)),
        "source_quality": _source_quality(assigned_sources),
        "tool_calls": direct_tool_calls,
        "dropped_claims": validation["dropped_claims"],
        "validation_gaps": validation_gaps,
        "researcher_error_status": researcher_error_status,
        "cell_id": task.get("cell_id", task["id"]),
        "entity": task.get("entity", ""),
        "dimension": task.get("dimension", ""),
    }
    log_event(
        "researcher",
        state.get("session_id", "-"),
        task_id=task["id"],
        mode="direct_fetch_fallback" if recovered_from_error else "direct_fetch",
        tool_calls=direct_tool_calls,
        urls_extracted=len(urls),
        claims_extracted=len(valid_claims),
        claims_raw=extract_stats["raw_claims"],
        claims_dropped_disallowed=extract_stats["dropped_disallowed_url"],
        claims_dropped_empty=extract_stats["dropped_empty"],
        claims_dropped_entity=extract_stats["dropped_entity_contamination"],
        claims_fallback=fallback_claims,
        claims_dropped_validation=len(validation["dropped_claims"]),
        researcher_error_status=researcher_error_status,
        failure_reason=failure_reason,
        top_source_domains=[
            domain
            for domain, _ in Counter(
                urlparse(u).netloc for u in urls if urlparse(u).netloc
            ).most_common(5)
        ],
        development=False,
    )
    return {
        "findings": [finding],
        "total_tool_calls": direct_tool_calls,
        "total_tokens_used": extract_tokens,
    }


def _extract_claims(
    messages, allowed_urls: list[str], task_question: str, user_query: str = ""
) -> tuple[list[Claim], str, int, dict[str, int]]:
    """Run a structured-output pass over the ReAct transcript.

    Returns (claims, narrative, tokens_used, stats).
    """
    stats = {
        "raw_claims": 0,
        "dropped_disallowed_url": 0,
        "dropped_empty": 0,
        "dropped_entity_contamination": 0,
    }
    if not messages:
        return [], "", 0, stats

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
        "- Return 5-12 claims, or fewer if evidence is thin. Empty list is fine.\n"
        "- Also write a 2-4 sentence `narrative` answering the sub-task."
    )

    tokens = 0
    try:
        def _call_extractor():
            llm = get_extractor_llm().with_structured_output(
                ResearcherStructuredOutput, include_raw=True
            )
            return llm.invoke(
                extractor_prompt,
                config={
                    "metadata": {
                        "prompt_version": PROMPT_VERSION,
                        "agent": "researcher_extract",
                    }
                },
            )

        out = invoke_with_rotation(
            "groq",
            _call_extractor,
            attempts=2,
            rotate=rotate_groq,
        )
        if isinstance(out, dict):
            tokens = tokens_from_response(out.get("raw"))
            if out.get("parsing_error") and not out.get("parsed"):
                return [], "", tokens, stats
            result = out.get("parsed")
        else:
            result = out
        if result is None:
            return [], "", tokens, stats
    except Exception:
        return [], "", tokens, stats

    claims: list[Claim] = []
    for c in result.claims or []:
        stats["raw_claims"] += 1
        url = (c.source_url or "").rstrip(".,);:")
        if url not in allowed:
            stats["dropped_disallowed_url"] += 1
            continue
        statement = (c.statement or "").strip()
        snippet = (c.snippet or "").strip()[:300]
        if not statement or not snippet:
            stats["dropped_empty"] += 1
            continue
        if is_entity_contaminated(user_query, f"{statement}\n{snippet}", url):
            stats["dropped_entity_contamination"] += 1
            continue
        claims.append(
            Claim(
                statement=statement,
                source_url=url,
                snippet=snippet,
                confidence=max(0.0, min(1.0, float(c.confidence))),
            )
        )
    return claims, (result.narrative or "").strip(), tokens, stats


def researcher_node(state: ResearcherState) -> dict:
    """Run the ReAct agent on one sub-task. Called via Send() in parallel."""
    task = state["task"]
    assigned_sources = state.get("assigned_sources", [])

    if is_development():
        finding: Finding = {
            "task_id": task["id"],
            "sub_question": task["question"],
            "answer": (
                f"Development-mode finding for '{task['question']}'. "
                "External LLM and tool calls are disabled."
            ),
            "content": (
                f"Development-mode finding for '{task['question']}'. "
                "External LLM and tool calls are disabled."
            ),
            "claims": [],
            "sources": [],
            "gaps": ["Development mode does not perform source-backed research."],
            "confidence": 0.5,
            "source_quality": 0.0,
            "tool_calls": 0,
        }
        log_event(
            "researcher",
            state.get("session_id", "-"),
            task_id=task["id"],
            tool_calls=0,
            urls_extracted=0,
            claims_extracted=0,
            claims_dropped_disallowed=0,
            claims_dropped_entity=0,
            top_source_domains=[],
            development=True,
        )
        return {"findings": [finding], "total_tool_calls": 0}

    system_msg = RESEARCHER_PROMPT.format(
        user_query=state["user_query"],
        question=task["question"],
        rationale=task["rationale"],
        assigned_sources=_format_assigned_sources(assigned_sources),
        max_steps=REACT_MAX_STEPS,
    )

    direct_update: dict | None = None
    required_evidence = int(task.get("required_evidence", 1) or 1)
    if assigned_sources:
        direct_update = _direct_fetch_finding(task, assigned_sources, state)
        direct_claims = (direct_update.get("findings") or [{}])[0].get("claims") or []
        if len(direct_claims) >= required_evidence:
            return direct_update
        tried = ", ".join((s.get("url") or s.get("canonical_url") or "") for s in assigned_sources[:4])
        system_msg += (
            "\n\n# Targeted ReAct Fallback\n"
            f"The direct-fetch pass found {len(direct_claims)}/{required_evidence} validated claims "
            f"for cell `{task.get('cell_id', task['id'])}`.\n"
            f"Cell entity: {task.get('entity', '') or '(none)'}\n"
            f"Cell dimension: {task.get('dimension', '') or '(general)'}\n"
            f"Success criteria: {'; '.join(task.get('success_criteria', []) or [])}\n"
            f"Already tried sources: {tried or '(none)'}\n"
            "Use tools only to fill this specific underfilled cell."
        )

    try:
        def _call_agent():
            return _get_agent().invoke(
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

        def _rotate(reason: str) -> None:
            rotate_groq(reason)
            _reset_agent()

        result = invoke_with_rotation(
            "groq",
            _call_agent,
            attempts=2,
            rotate=_rotate,
        )
    except Exception as e:
        err = f"researcher[{task['id']}]: {e}"
        if direct_update is not None:
            direct_update["errors"] = [err]
            finding = (direct_update.get("findings") or [{}])[0]
            if len(finding.get("claims") or []) >= required_evidence:
                finding["researcher_error_status"] = "recovered"
            return direct_update
        if assigned_sources:
            update = _direct_fetch_finding(
                task,
                assigned_sources,
                state,
                recovered_from_error=True,
                failure_reason=err,
            )
            update["errors"] = [err]
            return update
        finding: Finding = {
            "task_id": task["id"],
            "sub_question": task["question"],
            "answer": f"Research failed: {type(e).__name__}: {e}",
            "content": f"Research failed: {type(e).__name__}: {e}",
            "claims": [],
            "sources": [],
            "gaps": [f"Research failed for this sub-question: {type(e).__name__}: {e}"],
            "confidence": 0.0,
            "source_quality": _source_quality(assigned_sources),
            "tool_calls": 0,
            "dropped_claims": [],
            "validation_gaps": [],
            "researcher_error_status": "unrecovered",
        }
        log_event(
            "researcher",
            state.get("session_id", "-"),
            task_id=task["id"],
            tool_calls=0,
            urls_extracted=0,
            claims_extracted=0,
            failure_reason=err,
        )
        return {"findings": [finding], "errors": [err]}

    messages = result.get("messages", [])
    final_msg = messages[-1] if messages else None
    answer_text = getattr(final_msg, "content", "") or ""

    urls = _extract_urls(messages)
    domains = {urlparse(u).netloc for u in urls if urlparse(u).netloc}
    tool_calls = sum(len(getattr(m, "tool_calls", []) or []) for m in messages)
    react_tokens = tokens_from_messages(messages)

    claims, narrative, extract_tokens, extract_stats = _extract_claims(
        messages, urls, task["question"], state["user_query"]
    )
    validation = validate_claims(
        claims,
        user_query=state["user_query"],
        task_question=task["question"],
        cell_id=task.get("cell_id", task["id"]),
        entity=task.get("entity", ""),
        dimension=task.get("dimension", ""),
    )
    valid_claims = validation["valid_claims"]
    if direct_update is not None:
        direct_finding = (direct_update.get("findings") or [{}])[0]
        direct_claims = direct_finding.get("claims") or []
        merged_claims = direct_claims + [
            c
            for c in valid_claims
            if (c.get("statement"), c.get("source_url"))
            not in {(d.get("statement"), d.get("source_url")) for d in direct_claims}
        ]
        direct_finding["claims"] = merged_claims
        direct_finding["sources"] = list(
            dict.fromkeys((direct_finding.get("sources") or []) + urls)
        )
        direct_finding["tool_calls"] = int(direct_finding.get("tool_calls") or 0) + tool_calls
        direct_finding["researcher_error_status"] = (
            "recovered" if len(merged_claims) >= required_evidence else "unrecovered"
        )
        direct_update["total_tool_calls"] = direct_update.get("total_tool_calls", 0) + tool_calls
        direct_update["total_tokens_used"] = (
            direct_update.get("total_tokens_used", 0) + react_tokens + extract_tokens
        )
        return direct_update
    content = narrative or answer_text
    gaps = _validation_gap_texts(validation["gaps"])
    if not valid_claims:
        gaps.append(f"No source-backed claims extracted for: {task['question']}")

    finding: Finding = {
        "task_id": task["id"],
        "sub_question": task["question"],
        "answer": content,
        "content": content,
        "claims": valid_claims,
        "sources": urls,
        "gaps": gaps,
        "confidence": _confidence(len(urls), len(domains)),
        "source_quality": _source_quality(assigned_sources),
        "tool_calls": tool_calls,
        "dropped_claims": validation["dropped_claims"],
        "validation_gaps": validation["gaps"],
        "researcher_error_status": "none" if valid_claims else "unrecovered",
        "cell_id": task.get("cell_id", task["id"]),
        "entity": task.get("entity", ""),
        "dimension": task.get("dimension", ""),
    }

    top_domains = [
        domain
        for domain, _ in Counter(urlparse(u).netloc for u in urls if urlparse(u).netloc).most_common(5)
    ]
    log_event(
        "researcher",
        state.get("session_id", "-"),
        task_id=task["id"],
        mode="react",
        tool_calls=tool_calls,
        urls_extracted=len(urls),
        claims_extracted=len(valid_claims),
        claims_raw=extract_stats["raw_claims"],
        claims_dropped_disallowed=extract_stats["dropped_disallowed_url"],
        claims_dropped_empty=extract_stats["dropped_empty"],
        claims_dropped_entity=extract_stats["dropped_entity_contamination"],
        claims_dropped_validation=len(validation["dropped_claims"]),
        top_source_domains=top_domains,
        development=False,
    )
    return {
        "findings": [finding],
        "total_tool_calls": tool_calls,
        "total_tokens_used": react_tokens + extract_tokens,
    }
