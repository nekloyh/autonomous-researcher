"""Web search tool with DuckDuckGo primary (free), Tavily fallback."""
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from tavily import TavilyClient

from app.config import TAVILY_API_KEY

_tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
_cache: dict[str, str] = {}
_CACHE_MAX = 500


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
    if query in _cache:
        return _cache[query]

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

    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[query] = out
    return out
