"""Utilities for keeping critic and validator gaps actionable."""
from __future__ import annotations

import re

from app.entity_guard import entities_in_text

_SPACE_RE = re.compile(r"\s+")
_DIMENSION_TERMS: dict[str, tuple[str, ...]] = {
    "products": ("product", "service", "launch", "announce", "solution"),
    "partnerships": ("partner", "partnership", "collaboration", "alliance"),
    "infrastructure": ("infrastructure", "investment", "data center", "cloud", "gpu"),
    "market": ("market", "customer", "user", "mau", "adoption"),
    "financials": ("revenue", "profit", "loss", "financial", "growth"),
    "evidence": ("unsupported", "source-backed", "evidence", "claim"),
}


def gap_dimension(text: str) -> str:
    lower = (text or "").lower()
    for dimension, terms in _DIMENSION_TERMS.items():
        if any(term in lower for term in terms):
            return dimension
    return "general"


def normalize_gap_key(gap: dict | str) -> str:
    if isinstance(gap, dict):
        text = " ".join(
            str(gap.get(k, ""))
            for k in ("question", "reason", "origin_task_id", "priority")
            if gap.get(k)
        )
    else:
        text = str(gap)
    entities = ",".join(sorted(entities_in_text(text))) or "none"
    dimension = gap_dimension(text)
    years = ",".join(sorted(set(re.findall(r"\b20\d{2}\b", text)))) or "any"
    compact = _SPACE_RE.sub(" ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()
    # Keep a short normalized suffix so distinct questions in the same bucket do
    # not collapse too aggressively.
    suffix = " ".join(compact.split()[:14])
    return f"{entities}|{dimension}|{years}|{suffix}"


def dedupe_gaps(gaps: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for gap in gaps:
        question = str(gap.get("question", "")).strip()
        if not question:
            continue
        key = normalize_gap_key(gap)
        if key in seen:
            continue
        seen.add(key)
        out.append(gap)
    return out
