"""Tests for deterministic claim validation."""
from __future__ import annotations

from app.claim_validation import validate_claims


def test_claim_validation_drops_year_claim_without_direct_year_evidence():
    out = validate_claims(
        [
            {
                "statement": "FPT launched an AI service in 2024.",
                "source_url": "https://fpt.com/en/news/ai",
                "snippet": "FPT launched an AI service for enterprise customers.",
                "confidence": 0.8,
            }
        ],
        user_query="Compare VNG and FPT AI strategy in 2024",
        task_question="What AI products did FPT launch in 2024?",
    )

    assert out["valid_claims"] == []
    assert out["dropped_claims"][0]["validation_status"] == "dropped"
    assert "year-bearing claim" in out["dropped_claims"][0]["validation_warnings"][0]
    assert out["gaps"][0]["priority"] == "high"


def test_claim_validation_enriches_valid_claim_metadata():
    out = validate_claims(
        [
            {
                "statement": "FPT announced AI as a strategic direction in 2024.",
                "source_url": "https://fptsoftware.com/newsroom/news-and-press-releases/press-release/fpt-unveils-strategic-directions-all-in-on-ai-automotive-and-semiconductor",
                "snippet": "FPT announced its 2024-2026 strategic directions including Artificial Intelligence.",
                "confidence": 0.8,
            }
        ],
        user_query="Compare VNG and FPT AI strategy in 2024",
        task_question="What is FPT AI strategy in 2024?",
    )

    assert len(out["valid_claims"]) == 1
    claim = out["valid_claims"][0]
    assert claim["source_domain"] == "fptsoftware.com"
    assert claim["source_policy_tier"] == "preferred"
    assert claim["evidence_years"] == ["2024", "2026"]
    assert claim["attributed_entities"] == ["fpt"]
