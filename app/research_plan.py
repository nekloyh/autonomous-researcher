"""Helpers for structured research plans and cell coverage."""
from __future__ import annotations

import re
from collections import defaultdict

from app.gaps import gap_dimension

_SPACE_RE = re.compile(r"\s+")


def normalize_cell_id(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return cleaned[:80] or "cell"


def claim_cell_id(claim: dict) -> str:
    return str(claim.get("cell_id") or "").strip()


def task_cell_id(task: dict) -> str:
    return str(task.get("cell_id") or task.get("id") or "").strip()


def build_cell_coverage(research_plan: dict, findings: list[dict]) -> list[dict]:
    """Summarize validated claims by planner cell."""
    cells = research_plan.get("research_cells") or []
    counts: dict[str, int] = defaultdict(int)
    sources: dict[str, set[str]] = defaultdict(set)
    for finding in findings or []:
        fallback_cell = finding.get("cell_id") or finding.get("task_id")
        for claim in finding.get("claims") or []:
            if claim.get("validation_status", "valid") == "dropped":
                continue
            cell_id = claim.get("cell_id") or fallback_cell
            if not cell_id:
                continue
            counts[str(cell_id)] += 1
            if claim.get("source_url"):
                sources[str(cell_id)].add(claim["source_url"])

    coverage: list[dict] = []
    for cell in cells:
        cell_id = str(cell.get("id") or cell.get("cell_id") or "")
        required = int(cell.get("required_evidence") or 1)
        count = counts.get(cell_id, 0)
        status = "filled" if count >= required else "partial" if count else "unfilled"
        coverage.append(
            {
                **cell,
                "cell_id": cell_id,
                "validated_claims": count,
                "source_count": len(sources.get(cell_id, set())),
                "cell_status": status,
            }
        )
    return coverage


def underfilled_cells(research_plan: dict, findings: list[dict]) -> list[dict]:
    return [
        c
        for c in build_cell_coverage(research_plan, findings)
        if c.get("cell_status") != "filled" and not c.get("allow_insufficient_data")
    ]


def gap_to_cell(gap: dict, research_plan: dict) -> dict | None:
    """Best-effort mapping from a critic gap to an existing research cell."""
    text = " ".join(str(gap.get(k, "")) for k in ("question", "reason"))
    lower = _SPACE_RE.sub(" ", text.lower())
    dimension = gap_dimension(text)
    best: tuple[int, dict] | None = None
    for cell in research_plan.get("research_cells") or []:
        score = 0
        entity = str(cell.get("entity") or "").lower()
        cell_dim = str(cell.get("dimension") or "").lower()
        question = str(cell.get("question") or "").lower()
        if entity and entity in lower:
            score += 3
        if cell_dim and (cell_dim in lower or dimension in cell_dim):
            score += 3
        for token in set(re.findall(r"[a-z0-9]+", question)):
            if len(token) > 3 and token in lower:
                score += 1
        if score and (best is None or score > best[0]):
            best = (score, cell)
    return best[1] if best else None
