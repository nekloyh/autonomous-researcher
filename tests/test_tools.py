"""Unit tests for the four tools — external services are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_python_exec_basic_math():
    from app.tools.python_exec import python_exec

    out = python_exec.invoke({"code": "print(2 + 2)"})
    assert "4" in out


def test_python_exec_handles_error():
    from app.tools.python_exec import python_exec

    out = python_exec.invoke({"code": "raise ValueError('boom')"})
    assert "ERROR" in out or "ValueError" in out


def test_fetch_url_rejects_invalid_scheme():
    from app.tools.fetch_url import fetch_url

    out = fetch_url.invoke({"url": "ftp://example.com"})
    assert "ERROR" in out and "Invalid URL" in out


@patch("app.tools.web_search._tavily")
def test_web_search_uses_tavily(mock_tavily):
    from app.tools.web_search import _cache, web_search

    _cache.clear()
    mock_tavily.search.return_value = {
        "results": [
            {"title": "Test", "url": "https://example.com", "content": "snippet"}
        ]
    }
    out = web_search.invoke({"query": "anything unique 12345"})
    assert "example.com" in out
    assert "[1]" in out


@patch("app.tools.web_search._tavily")
def test_web_search_falls_back_to_ddg(mock_tavily):
    from app.tools.web_search import _cache, web_search

    _cache.clear()
    mock_tavily.search.side_effect = RuntimeError("rate limit")
    with patch("app.tools.web_search.DDGS") as mock_ddgs:
        instance = MagicMock()
        instance.text.return_value = [
            {"title": "Fallback", "href": "https://fb.example", "body": "..."}
        ]
        mock_ddgs.return_value = instance
        out = web_search.invoke({"query": "rare-fallback-query-xyz"})
    assert "fb.example" in out


def test_vector_search_handles_missing_collection():
    from app.tools.vector_search import vector_search

    with patch("app.tools.vector_search._get_client") as mock_client:
        client = MagicMock()
        client.collection_exists.return_value = False
        mock_client.return_value = client
        out = vector_search.invoke({"query": "anything"})
    assert "No documents indexed" in out


@pytest.mark.parametrize("name", ["web_search", "fetch_url", "vector_search", "python_exec"])
def test_all_tools_have_docstrings(name):
    """LLM relies on the docstring to choose tools — they must be non-empty."""
    from app.tools import ALL_TOOLS

    tool = next(t for t in ALL_TOOLS if t.name == name)
    assert tool.description and len(tool.description) > 50
