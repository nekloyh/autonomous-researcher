"""Lightweight observability helpers.

The token accounting is best-effort: LangChain providers expose
`usage_metadata` on AIMessage objects with `{"input_tokens", "output_tokens",
"total_tokens"}`. We read it when present and return 0 otherwise — never raise.
"""
from __future__ import annotations

from typing import Any


def _usage_total(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    if "total_tokens" in usage:
        try:
            return int(usage["total_tokens"])
        except (TypeError, ValueError):
            return 0
    try:
        return int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
    except (TypeError, ValueError):
        return 0


def tokens_from_response(response: Any) -> int:
    """Tokens from a single AIMessage-like response."""
    return _usage_total(getattr(response, "usage_metadata", None))


def tokens_from_messages(messages: list) -> int:
    """Tokens from a ReAct agent's message list."""
    total = 0
    for m in messages or []:
        total += _usage_total(getattr(m, "usage_metadata", None))
    return total
