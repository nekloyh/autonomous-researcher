"""Tests for entity contamination guard behavior."""
from __future__ import annotations

from app.entity_guard import is_entity_contaminated


def test_comparison_query_allows_each_queried_entity_individually():
    query = "MoMo vs ZaloPay vs VNPay market share in 2024"

    assert not is_entity_contaminated(
        query,
        "ZaloPay expanded QR payment coverage in Vietnam.",
        "https://zalopay.vn/news",
    )
    assert not is_entity_contaminated(
        query,
        "VNPay reported merchant payment services in Vietnam.",
        "https://vnpay.vn/about",
    )


def test_comparison_query_blocks_entities_outside_query():
    query = "VNG vs FPT AI strategy in 2024"

    assert is_entity_contaminated(
        query,
        "Tiki launched a retail logistics initiative.",
        "https://tiki.vn/news",
    )


def test_single_entity_query_still_blocks_other_known_entities():
    assert is_entity_contaminated(
        "MoMo business model",
        "VNG operates online games and digital services.",
        "https://vng.com.vn/about",
    )
