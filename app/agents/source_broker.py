"""Source broker: deterministic URL discovery, ranking, and deduplication."""
from __future__ import annotations

import re
from collections import defaultdict
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.config import MAX_SOURCES_PER_TASK, SOURCE_BROKER_SEARCH_RESULTS, is_development
from app.logger import log_event
from app.state import AgentState, SourceCandidate, SubTask
from app.tools.web_search import search_web_results

_TRACKING_PREFIXES = ("utm_",)
_TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src"}
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP_TERMS = {
    "about",
    "analysis",
    "business",
    "company",
    "compare",
    "current",
    "latest",
    "market",
    "model",
    "news",
    "official",
    "press",
    "release",
    "report",
    "strategy",
    "the",
    "what",
    "with",
}
_REPUTABLE_MEDIA = {
    "reuters.com",
    "bloomberg.com",
    "apnews.com",
    "ft.com",
    "wsj.com",
    "cnbc.com",
    "techcrunch.com",
    "dealstreetasia.com",
    "techinasia.com",
}
_DATABASE_DOMAINS = {
    "crunchbase.com",
    "tracxn.com",
    "cbinsights.com",
    "pitchbook.com",
}
_GENERIC_DOMAINS = {
    "wikipedia.org",
    "linkedin.com",
    "facebook.com",
    "youtube.com",
    "reddit.com",
    "medium.com",
}


def canonicalize_url(url: str) -> str:
    """Normalize URLs so search duplicates collapse before fan-out."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lower = key.lower()
        if lower in _TRACKING_KEYS or lower.startswith(_TRACKING_PREFIXES):
            continue
        query_items.append((key, value))
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower().removeprefix("www."),
            path,
            "",
            urlencode(query_items, doseq=True),
            "",
        )
    )


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _terms(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall((text or "").lower()) if len(t) > 2 and t not in _STOP_TERMS}


def classify_source(url: str, title: str = "", snippet: str = "", query: str = "") -> str:
    domain = _domain(url)
    text = f"{title} {snippet} {domain}".lower()
    query_terms = _terms(query)
    if any(domain == d or domain.endswith("." + d) for d in _REPUTABLE_MEDIA):
        return "reputable_media"
    if any(domain == d or domain.endswith("." + d) for d in _DATABASE_DOMAINS):
        return "database"
    if any(domain == d or domain.endswith("." + d) for d in _GENERIC_DOMAINS):
        return "generic"
    if query_terms and any(term in domain for term in query_terms):
        return "official"
    if "official" in text or "press release" in text or "investor" in text:
        return "official"
    return "unknown"


def rank_source(query: str, result: dict) -> float:
    title = result.get("title") or ""
    snippet = result.get("content") or result.get("snippet") or ""
    url = result.get("url") or ""
    domain = _domain(url)
    text_terms = _terms(f"{title} {snippet} {domain}")
    overlap = len(_terms(query) & text_terms)
    source_type = classify_source(url, title, snippet, query)
    score = float(overlap)
    score += {
        "official": 7.0,
        "reputable_media": 5.0,
        "database": 4.0,
        "unknown": 0.0,
        "generic": -4.0,
    }[source_type]
    if "annual report" in f"{title} {snippet}".lower():
        score += 3.0
    if "press release" in f"{title} {snippet}".lower():
        score += 2.0
    return score


def _candidate(query: str, result: dict, task_id: str) -> SourceCandidate | None:
    url = (result.get("url") or "").strip()
    canonical = canonicalize_url(url)
    if not canonical:
        return None
    title = (result.get("title") or "").strip()
    snippet = (result.get("content") or result.get("snippet") or "").strip()
    source_type = classify_source(canonical, title, snippet, query)
    return {
        "url": url,
        "canonical_url": canonical,
        "title": title,
        "snippet": snippet[:500],
        "domain": _domain(canonical),
        "rank_score": rank_source(query, result),
        "source_type": source_type,  # type: ignore[typeddict-item]
        "assigned_task_ids": [task_id],
    }


def _search_query(user_query: str, task: SubTask) -> str:
    question = task.get("question", "")
    if user_query.lower() in question.lower():
        return question
    return f"{question} {user_query}".strip()


def source_broker_node(state: AgentState) -> dict:
    """Discover ranked source candidates for the current plan without LLM calls."""
    plan = state.get("plan", []) or []
    if not plan:
        return {"source_candidates": []}
    if is_development():
        log_event(
            "source_broker",
            state.get("session_id", "-"),
            tasks=len(plan),
            searches=0,
            candidates=0,
            development=True,
        )
        return {"source_candidates": []}

    by_canonical: dict[str, SourceCandidate] = {}
    per_task_counts: dict[str, int] = defaultdict(int)
    per_task_domain_counts: dict[tuple[str, str], int] = defaultdict(int)
    errors: list[str] = []
    searches = 0

    for task in plan:
        task_id = task["id"]
        query = _search_query(state["user_query"], task)
        try:
            results = search_web_results(query, max_results=SOURCE_BROKER_SEARCH_RESULTS)
            searches += 1
        except Exception as e:
            errors.append(f"source_broker[{task_id}]: {type(e).__name__}: {e}")
            continue

        ranked = sorted(results, key=lambda r: rank_source(query, r), reverse=True)
        for result in ranked:
            cand = _candidate(query, result, task_id)
            if cand is None:
                continue
            canonical = cand["canonical_url"]
            domain = cand.get("domain", "")
            existing = by_canonical.get(canonical)
            if existing is not None:
                ids = existing.setdefault("assigned_task_ids", [])
                if task_id not in ids:
                    ids.append(task_id)
                continue
            if per_task_counts[task_id] >= MAX_SOURCES_PER_TASK:
                continue
            if domain and per_task_domain_counts[(task_id, domain)] >= 2:
                continue
            by_canonical[canonical] = cand
            per_task_counts[task_id] += 1
            if domain:
                per_task_domain_counts[(task_id, domain)] += 1

    candidates = sorted(
        by_canonical.values(),
        key=lambda c: (float(c.get("rank_score", 0.0)), c.get("domain", "")),
        reverse=True,
    )
    log_event(
        "source_broker",
        state.get("session_id", "-"),
        tasks=len(plan),
        searches=searches,
        candidates=len(candidates),
        errors=len(errors),
    )
    update: dict = {"source_candidates": candidates, "total_tool_calls": searches}
    if errors:
        update["errors"] = errors
    return update
