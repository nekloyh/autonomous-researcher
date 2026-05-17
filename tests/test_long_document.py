"""Tests for targeted long-document section extraction."""
from __future__ import annotations

from app.long_document import relevant_document_excerpt


def test_relevant_document_excerpt_ranks_keyword_chunks(monkeypatch):
    long_text = (
        "### UNTRUSTED WEB CONTENT BELOW\n\n"
        "## General\nThis section discusses generic company history.\n"
        + ("generic text " * 300)
        + "\n## AI Partnerships\nFPT announced AI partnerships with technology companies in 2024.\n"
        + ("FPT AI partnership NVIDIA Samsung university " * 80)
    )

    class FakeFetch:
        def invoke(self, args):  # noqa: ARG002
            return long_text

    monkeypatch.setattr("app.long_document.fetch_url", FakeFetch())

    excerpt = relevant_document_excerpt(
        "https://fpt.com/annual-report",
        ["FPT AI partnerships 2024 NVIDIA"],
        max_chars=3000,
    )

    assert "AI Partnerships" in excerpt
    assert "NVIDIA" in excerpt
