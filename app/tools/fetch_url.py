
"""Fetch full content of a URL as markdown."""
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from markdownify import markdownify
from tenacity import retry, stop_after_attempt, wait_exponential


@tool
@retry(stop=stop_after_attempt(2), wait=wait_exponential(max=10))
def fetch_url(url: str) -> str:
    """Fetch and convert a webpage to markdown.

    Use this after web_search when you need the FULL content of a specific page.

    Args:
        url: The full URL to fetch (must start with http:// or https://)

    Returns:
        Markdown-formatted content of the page, truncated to 8000 chars.
        Returns error message if fetch fails.
    """
    if not url.startswith(("http://", "https://")):
        return f"ERROR: Invalid URL '{url}'. Must start with http:// or https://"

    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (ResearchAgent/1.0)"},
        )
        resp.raise_for_status()
    except requests.Timeout:
        return f"ERROR: Timeout fetching {url}. Try a different source."
    except requests.HTTPError as e:
        return f"ERROR: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"ERROR: Could not fetch {url}. Reason: {e}"

    # Parse HTML
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "aside", "iframe"]):
        tag.decompose()

    # Convert to markdown
    main = soup.find("main") or soup.find("article") or soup.body or soup
    md = markdownify(str(main), heading_style="ATX")

    # Truncate
    max_chars = 8000
    if len(md) > max_chars:
        return md[:max_chars] + f"\n\n[TRUNCATED. Full length: {len(md)} chars]"
    return md
