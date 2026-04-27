"""Web search tool with Tavily primary, DuckDuckGo fallback."""
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from tavily import TavilyClient

from app.config import TAVILY_API_KEY

_tavily = TavilyClient(api_key=TAVILY_API_KEY)
_cache: dict[str, str] = {}
_CACHE_MAX = 500


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

    try:
        resp = _tavily.search(query=query, search_depth="basic", max_results=5)
        results = resp.get("results", [])
    except Exception as e:
        try:
            ddgs = DDGS()
            raw = list(ddgs.text(query, max_results=5))
            results = [
                {"title": r["title"], "url": r["href"], "content": r["body"]}
                for r in raw
            ]
        except Exception as e2:
            return f"ERROR: Both search providers failed. Tavily: {e}. DDG: {e2}"

    if not results:
        return f"No results found for: {query}. Try different keywords."

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r['title']}\n"
            f"    URL: {r['url']}\n"
            f"    {r.get('content', '')[:300]}..."
        )
    out = "\n\n".join(formatted)

    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[query] = out
    return out
