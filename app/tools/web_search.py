"""Web search tool with provider fallback and small in-memory cache."""
from __future__ import annotations

import re
import time

from ddgs import DDGS
from langchain_core.tools import tool
from tavily import TavilyClient

from app.config import SEARCH_PROVIDER_ORDER
from app.provider_rotation import TAVILY_KEYS, is_limit_error, rotate_tavily

_tavily = None  # Test hook; production builds clients from the current key ring.

# Cache entries: query -> (stored_at_unix, formatted_result)
_cache: dict[str, tuple[float, str]] = {}
_CACHE_MAX = 500
_CACHE_TTL_SECONDS = 3600  # 1 hour

# If the query references a time horizon, skip cache so we don't serve stale data.
_TIME_SENSITIVE_RE = re.compile(
    r"\b(20\d{2}|Q[1-4]|today|latest|current|this (week|month|year)|recent|now)\b",
    re.IGNORECASE,
)
_BUSINESS_QUERY_RE = re.compile(
    r"\b(company|business model|revenue|valuation|funding|investor|users?|mau|"
    r"growth|market share|partnership|press release|annual report|quarterly)\b",
    re.IGNORECASE,
)
_STOP_TERMS = {
    "about",
    "business",
    "company",
    "current",
    "latest",
    "model",
    "news",
    "official",
    "press",
    "release",
    "report",
    "the",
    "what",
    "with",
}
_REPUTABLE_DOMAINS = (
    "reuters.com",
    "bloomberg.com",
    "apnews.com",
    "ft.com",
    "wsj.com",
    "cnbc.com",
    "forbes.com",
    "techcrunch.com",
    "dealstreetasia.com",
    "techinasia.com",
    "crunchbase.com",
    "tracxn.com",
    "cbinsights.com",
    "pitchbook.com",
)
_GENERIC_DOMAINS = (
    "wikipedia.org",
    "linkedin.com",
    "facebook.com",
    "youtube.com",
    "pinterest.",
    "reddit.com",
    "medium.com",
)


def _is_time_sensitive(query: str) -> bool:
    return bool(_TIME_SENSITIVE_RE.search(query))


def _domain(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).netloc.lower().removeprefix("www.")


def _query_terms(query: str) -> set[str]:
    terms = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", query or "") if len(t) > 2}
    return {t for t in terms if t not in _STOP_TERMS}


def _is_business_query(query: str) -> bool:
    return bool(_BUSINESS_QUERY_RE.search(query))


def _quality_score(query: str, result: dict) -> int:
    domain = _domain(result.get("url", ""))
    title = (result.get("title") or "").lower()
    content = (result.get("content") or "").lower()
    text = f"{title} {content} {domain}"
    terms = _query_terms(query)
    overlap = len(terms & set(re.findall(r"[a-z0-9]+", text)))

    score = overlap
    if terms and any(term in domain for term in terms):
        score += 6  # likely official or entity-specific domain
    if any(domain == d or domain.endswith("." + d) for d in _REPUTABLE_DOMAINS):
        score += 5
    if "press release" in text or "/news" in result.get("url", "").lower():
        score += 3
    if "investor" in text or "funding" in text or "annual report" in text:
        score += 3
    if any(g in domain for g in _GENERIC_DOMAINS):
        score -= 4
    if terms and overlap == 0:
        score -= 6
    return score


def _rank_results(query: str, results: list[dict]) -> list[dict]:
    if not _is_business_query(query):
        return results
    indexed = list(enumerate(results))
    indexed.sort(key=lambda item: (_quality_score(query, item[1]), -item[0]), reverse=True)
    return [item for _, item in indexed]


def _format_results(results: list) -> str:
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r['title']}\n"
            f"    URL: {r['url']}\n"
            f"    {r.get('content', '')[:300]}..."
        )
    return "\n\n".join(formatted)


def _search_ddg(query: str, max_results: int) -> list[dict]:
    ddgs = DDGS()
    raw = list(ddgs.text(query, max_results=max_results))
    return [
        {"title": r["title"], "url": r["href"], "content": r["body"]}
        for r in raw
    ]


def _tavily_client():
    if _tavily is not None:
        return _tavily
    key = TAVILY_KEYS.current()
    if not key:
        return None
    return TavilyClient(api_key=key)


def _search_tavily(query: str, max_results: int) -> list[dict]:
    client = _tavily_client()
    if client is None:
        raise RuntimeError("TAVILY_API_KEY/TAVILY_API_KEYS is not configured")
    try:
        resp = client.search(query=query, search_depth="basic", max_results=max_results)
    except Exception as exc:
        if is_limit_error(exc) and TAVILY_KEYS.has_multiple():
            rotate_tavily(f"{type(exc).__name__}: {exc}")
            client = _tavily_client()
            if client is not None:
                resp = client.search(query=query, search_depth="basic", max_results=max_results)
            else:
                raise
        else:
            raise
    return resp.get("results", [])


def search_web_results(query: str, max_results: int = 10) -> list[dict]:
    """Return ranked raw search results for deterministic non-LLM nodes."""
    errors: list[str] = []
    providers = SEARCH_PROVIDER_ORDER or ["tavily", "ddg"]
    for provider in providers:
        try:
            if provider == "tavily":
                results = _search_tavily(query, max_results)
            elif provider in {"ddg", "duckduckgo"}:
                results = _search_ddg(query, max_results)
            else:
                errors.append(f"{provider}: unknown provider")
                continue
            if results:
                return _rank_results(query, results)
        except Exception as exc:
            errors.append(f"{provider}: {type(exc).__name__}: {exc}")
            continue
    raise RuntimeError("All search providers failed. " + " | ".join(errors))


@tool
def web_search(query: str) -> str:
    """Search the web for current information.

    Use this when you need:
    - Recent news or events
    - Current statistics, prices, or data
    - Information about specific companies, people, or products

    Args:
        query: Search query. Be specific (include dates, names, context).
            GOOD: "VNG Corporation Q3 2024 revenue report"
            BAD: "vietnamese companies"

    Returns:
        Formatted string with top 5 results: title, URL, snippet.
    """
    now = time.time()
    time_sensitive = _is_time_sensitive(query)

    if not time_sensitive:
        cached = _cache.get(query)
        if cached is not None:
            stored_at, value = cached
            if now - stored_at <= _CACHE_TTL_SECONDS:
                return value
            # Stale → drop and re-fetch below
            _cache.pop(query, None)

    try:
        results = search_web_results(query, max_results=10)
    except Exception as e:
        return f"ERROR: {e}"

    if not results:
        return f"No results found for: {query}. Try different keywords."

    out = _format_results(_rank_results(query, results)[:5])

    if not time_sensitive:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[query] = (now, out)
    return out
