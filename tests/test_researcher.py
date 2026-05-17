"""Tests for researcher extraction fallbacks."""
from __future__ import annotations

from app.agents.researcher import _fallback_claims_from_sources


def test_fallback_claims_from_sources_uses_allowed_fetched_urls():
    sources = [
        {
            "url": "https://theinvestor.vn/vietnam-tech-unicorn-vng-merges-two-core-units-into-ai-focused-greennode-brand-d17880.html",
            "title": "Vietnam tech unicorn VNG merges units into AI-focused GreenNode",
            "snippet": "VNG merged VNG Cloud with AI infrastructure unit GreenNode to form an AI-focused brand.",
            "source_type": "reputable_media",
        },
        {
            "url": "https://facebook.com/vng",
            "title": "VNG social profile",
            "snippet": "Generic profile page.",
            "source_type": "generic",
            "source_policy_tier": "blocked",
        },
    ]

    claims = _fallback_claims_from_sources(
        sources,
        [
            "https://theinvestor.vn/vietnam-tech-unicorn-vng-merges-two-core-units-into-ai-focused-greennode-brand-d17880.html",
            "https://facebook.com/vng",
        ],
        "Compare VNG and FPT AI strategy in 2024",
    )

    assert len(claims) == 1
    assert claims[0]["source_url"].startswith("https://theinvestor.vn/")
    assert "VNG" in claims[0]["statement"]
    assert claims[0]["confidence"] == 0.55


def test_fallback_claims_from_sources_drops_blocked_policy_sources():
    claims = _fallback_claims_from_sources(
        [
            {
                "url": "https://scribd.com/document/vng-ai",
                "title": "VNG AI mirror",
                "snippet": "VNG AI strategy in 2024.",
                "source_type": "unknown",
                "source_policy_tier": "blocked",
            }
        ],
        ["https://scribd.com/document/vng-ai"],
        "Compare VNG and FPT AI strategy in 2024",
    )

    assert claims == []


def test_fallback_claims_from_sources_respects_entity_guard():
    claims = _fallback_claims_from_sources(
        [
            {
                "url": "https://tiki.vn/news",
                "title": "Tiki announces logistics update",
                "snippet": "Tiki expanded logistics services in Vietnam.",
                "source_type": "unknown",
            }
        ],
        ["https://tiki.vn/news"],
        "MoMo business model",
    )

    assert claims == []
