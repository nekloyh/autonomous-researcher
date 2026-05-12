"""Tests for the deterministic citation map / renumber pipeline."""
from __future__ import annotations

from app.agents.synthesizer import (
    _append_sources_section,
    _build_citation_map,
    _renumber_and_filter,
)


def _finding(task_id: str, claims: list[dict], sources: list[str] | None = None) -> dict:
    return {
        "task_id": task_id,
        "content": f"narrative for {task_id}",
        "claims": claims,
        "sources": sources or [c["source_url"] for c in claims],
        "confidence": 0.8,
        "tool_calls": 2,
    }


def test_citation_map_dedupes_and_orders():
    findings = [
        _finding(
            "t1",
            [
                {"statement": "x", "source_url": "https://a.example", "snippet": "s1", "confidence": 0.9},
                {"statement": "y", "source_url": "https://b.example", "snippet": "s2", "confidence": 0.8},
            ],
        ),
        _finding(
            "t2",
            [
                # duplicate URL must reuse same number
                {"statement": "z", "source_url": "https://a.example", "snippet": "s3", "confidence": 0.7},
                {"statement": "w", "source_url": "https://c.example", "snippet": "s4", "confidence": 0.7},
            ],
        ),
    ]
    cmap = _build_citation_map(findings)
    assert cmap["https://a.example"] == 1
    assert cmap["https://b.example"] == 2
    assert cmap["https://c.example"] == 3
    assert len(cmap) == 3


def test_renumber_drops_orphan_brackets():
    cmap = {"https://a.example": 1, "https://b.example": 2}
    draft = (
        "Claim one [1]. Claim two [2]. Hallucinated [7] should disappear. Repeat [1]."
    )
    out, new_cmap = _renumber_and_filter(draft, cmap)
    assert "[7]" not in out
    assert "[1]" in out and "[2]" in out
    # Original numbering happens to already be 1,2 so renumber is a no-op here.
    assert new_cmap["https://a.example"] == 1
    assert new_cmap["https://b.example"] == 2


def test_renumber_compacts_after_drop():
    cmap = {"https://a.example": 1, "https://b.example": 2, "https://c.example": 3}
    # Draft only references 2 and 3 → after renumber they should become 1 and 2.
    draft = "First fact [2]. Second fact [3]. Unsupported [9]."
    out, new_cmap = _renumber_and_filter(draft, cmap)
    assert "[9]" not in out
    assert new_cmap == {"https://b.example": 1, "https://c.example": 2}


def test_sources_section_appended_at_end():
    cmap = {"https://a.example": 1, "https://b.example": 2}
    body = "# Title\n\n## 1. Section\nClaim [1]. Other [2].\n"
    out = _append_sources_section(body, cmap)
    assert "## Sources" in out
    # Sources must be in numeric order
    a_idx = out.index("[1] https://a.example")
    b_idx = out.index("[2] https://b.example")
    assert a_idx < b_idx


def test_sources_section_replaces_existing():
    cmap = {"https://real.example": 1}
    body = (
        "# Title\n\n## 1. Section\nClaim [1].\n\n## Sources\n[1] https://fake.example\n"
    )
    out = _append_sources_section(body, cmap)
    assert "https://fake.example" not in out
    assert "https://real.example" in out
