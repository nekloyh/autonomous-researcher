"""Web search tool with DuckDuckGo primary (free), Tavily fallback."""
from __future__ import annotations

import re
import time

from duckduckgo_search import DDGS
from langchain_core.tools import tool
from tavily import TavilyClient

from app.config import TAVILY_API_KEY

_tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# Cache entries: query -> (stored_at_unix, formatted_result)
_cache: dict[str, tuple[float, str]] = {}
_CACHE_MAX = 500
_CACHE_TTL_SECONDS = 3600  # 1 hour

# If the query references a time horizon, skip cache so we don't serve stale data.
_TIME_SENSITIVE_RE = re.compile(
    r"\b(20\d{2}|Q[1-4]|today|latest|current|this (week|month|year)|recent|now)\b",
    re.IGNORECASE,
)


def _is_time_sensitive(query: str) -> bool:
    return bool(_TIME_SENSITIVE_RE.search(query))


def _format_results(results: list) -> str:
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r['title']}\n"
            f"    URL: {r['url']}\n"
            f"    {r.get('content', '')[:300]}..."
        )
    return "\n\n".join(formatted)


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

    # DuckDuckGo primary (free, unlimited)
    try:
        ddgs = DDGS()
        raw = list(ddgs.text(query, max_results=5))
        results = [
            {"title": r["title"], "url": r["href"], "content": r["body"]}
            for r in raw
        ]
    except Exception as ddg_err:
        # Tavily fallback (paid quota — only hits when DDG fails)
        if _tavily is None:
            return f"ERROR: DuckDuckGo failed and TAVILY_API_KEY is not configured. DDG: {ddg_err}"
        try:
            resp = _tavily.search(query=query, search_depth="basic", max_results=5)
            results = resp.get("results", [])
        except Exception as tavily_err:
            return f"ERROR: Both search providers failed. DDG: {ddg_err}. Tavily: {tavily_err}"

    if not results:
        return f"No results found for: {query}. Try different keywords."

    out = _format_results(results)

    if not time_sensitive:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[query] = (now, out)
    return out
