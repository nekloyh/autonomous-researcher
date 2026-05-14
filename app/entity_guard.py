"""Helpers to avoid mixing evidence across similarly discussed entities."""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")

# Keep this intentionally small and explicit. The guard is a safety net for
# known benchmark entities, not a general-purpose NER system.
_ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "momo": ("momo", "m_service"),
    "vng": ("vng", "vng corporation", "vnggames"),
    "zalopay": ("zalopay", "zalo pay"),
    "vnpay": ("vnpay", "vn pay"),
    "fpt": ("fpt",),
    "tiki": ("tiki",),
}


def _norm(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def entities_in_text(text: str) -> set[str]:
    """Return known entities explicitly mentioned in text."""
    haystack = f" {_norm(text)} "
    found: set[str] = set()
    for entity, aliases in _ENTITY_ALIASES.items():
        for alias in aliases:
            if f" {_norm(alias)} " in haystack:
                found.add(entity)
                break
    return found


def primary_entity(query: str) -> str | None:
    """Pick the first known entity mentioned in the user query."""
    found = entities_in_text(query)
    if not found:
        return None
    query_norm = f" {_norm(query)} "
    positions: list[tuple[int, str]] = []
    for entity, aliases in _ENTITY_ALIASES.items():
        for alias in aliases:
            idx = query_norm.find(f" {_norm(alias)} ")
            if idx >= 0:
                positions.append((idx, entity))
                break
    if not positions:
        return sorted(found)[0]
    return sorted(positions)[0][1]


def is_entity_contaminated(query: str, text: str, source_url: str = "") -> bool:
    """True when evidence discusses another known entity without the query entity.

    Relation evidence is allowed when the evidence itself names both entities.
    For example, a MoMo query may use a VNG source only if the claim/snippet also
    names MoMo.
    """
    primary = primary_entity(query)
    if not primary:
        return False

    mentioned = entities_in_text(f"{text}\n{source_url}")
    if not mentioned:
        return False
    if primary in mentioned:
        return False
    return bool(mentioned - {primary})
