"""Deterministic claim validation before claims can drive citations."""
from __future__ import annotations

from typing import TypedDict

from app.entity_guard import (
    entities_in_text,
    is_comparison_query,
    is_entity_contaminated,
    primary_entity,
)
from app.source_policy import (
    classify_policy_tier,
    domain_from_url,
    entity_from_domain,
    years_in_text,
)


class ClaimValidationResult(TypedDict):
    valid_claims: list[dict]
    dropped_claims: list[dict]
    gaps: list[dict]


def _source_index(source_candidates: list[dict]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for source in source_candidates or []:
        for key in ("url", "canonical_url"):
            url = (source.get(key) or "").rstrip(".,);:")
            if url:
                index[url] = source
    return index


def _attributed_entities(claim: dict, domain: str) -> list[str]:
    entities = set(
        entities_in_text(
            f"{claim.get('statement', '')} {claim.get('snippet', '')} {claim.get('raw_snippet', '')}"
        )
    )
    domain_entity = entity_from_domain(domain)
    if domain_entity:
        entities.add(domain_entity)
    return sorted(entities)


def _gap(question: str, reason: str, priority: str = "high") -> dict:
    return {
        "question": question,
        "origin_task_id": "claim_validation",
        "reason": reason,
        "priority": priority,
    }


def validate_claims(
    claims: list[dict],
    *,
    user_query: str,
    task_question: str = "",
    source_candidates: list[dict] | None = None,
    cell_id: str = "",
    entity: str = "",
    dimension: str = "",
    evidence_type: str = "factual",
) -> ClaimValidationResult:
    """Return valid claims plus dropped-claim diagnostics.

    The function accepts plain dicts so it can validate both TypedDict claims and
    Pydantic model dumps without coupling agent code to a concrete class.
    """
    source_by_url = _source_index(source_candidates or [])
    valid: list[dict] = []
    dropped: list[dict] = []
    gaps: list[dict] = []
    query_entities = entities_in_text(user_query)
    query_primary = primary_entity(user_query)

    for claim in claims or []:
        url = (claim.get("source_url") or "").rstrip(".,);:")
        source = source_by_url.get(url, {})
        snippet = (claim.get("raw_snippet") or claim.get("snippet") or "").strip()
        domain = domain_from_url(url)
        policy_tier = source.get("source_policy_tier") or classify_policy_tier(
            url,
            source.get("title", ""),
            snippet or source.get("snippet", ""),
            user_query,
        )
        evidence_years = sorted(years_in_text(f"{snippet} {source.get('title', '')}"))
        statement_years = years_in_text(claim.get("statement", ""))
        attributed = _attributed_entities(claim, domain)
        warnings: list[str] = []

        if policy_tier == "blocked":
            warnings.append(f"blocked source domain: {domain}")
        if is_entity_contaminated(user_query, f"{claim.get('statement', '')}\n{snippet}", url):
            warnings.append("entity contamination")
        if is_comparison_query(user_query):
            if query_entities and not (set(attributed) & query_entities):
                warnings.append("claim has no clear attribution to compared entities")
            elif set(attributed) - query_entities:
                warnings.append("claim attributes evidence to an entity outside the comparison")
        elif query_primary and attributed and query_primary not in attributed:
            warnings.append(f"claim is attributed to {', '.join(attributed)}, not {query_primary}")
        if statement_years and not (statement_years & set(evidence_years)):
            warnings.append(
                "year-bearing claim lacks direct year evidence in source snippet"
            )

        enriched = {
            **claim,
            "source_url": url,
            "source_domain": domain,
            "source_type": source.get("source_type") or claim.get("source_type") or "unknown",
            "source_policy_tier": policy_tier,
            "evidence_years": evidence_years,
            "raw_snippet": snippet,
            "attributed_entities": attributed,
            "validation_warnings": warnings,
            "validation_status": "dropped" if warnings else "valid",
            "cell_id": cell_id or claim.get("cell_id", ""),
            "entity": entity or claim.get("entity", ""),
            "dimension": dimension or claim.get("dimension", ""),
            "evidence_type": evidence_type or claim.get("evidence_type", "factual"),
            "document_section": claim.get("document_section", ""),
            "page_or_chunk": claim.get("page_or_chunk", ""),
        }
        if warnings:
            dropped.append(enriched)
            gaps.append(
                _gap(
                    f"Find source-backed evidence for: {task_question or user_query}",
                    "; ".join(warnings),
                    "high",
                )
            )
        else:
            valid.append(enriched)

    return {"valid_claims": valid, "dropped_claims": dropped, "gaps": gaps}
