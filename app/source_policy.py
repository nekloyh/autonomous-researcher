"""Source quality policy for research inputs.

This module is deliberately deterministic. The LLM can choose what to read, but
the pipeline decides which sources are eligible to become evidence.
"""
from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from app.entity_guard import entities_in_text

SourcePolicyTier = Literal["blocked", "preferred", "allowed"]
YearStatus = Literal["matched", "unknown", "mismatch"]

_YEAR_RE = re.compile(r"\b20\d{2}\b")

BLOCKED_DOMAINS = {
    "scribd.com",
    "academia.edu",
    "slideshare.net",
    "facebook.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "drive.google.com",
    "docs.google.com",
    "dropbox.com",
    "mediafire.com",
}

PREFERRED_ENTITY_DOMAINS: dict[str, tuple[str, ...]] = {
    "vng": ("vng.com.vn", "bctn2024.vng.com.vn"),
    "fpt": ("fpt.com", "fptsoftware.com", "investor.fpt.com"),
    "momo": ("momo.vn",),
    "zalopay": ("zalopay.vn",),
    "vnpay": ("vnpay.vn",),
    "tiki": ("tiki.vn",),
}

PREFERRED_MEDIA_DOMAINS = {
    "tuoitre.vn",
    "news.tuoitre.vn",
    "vnexpress.net",
    "cafef.vn",
    "theinvestor.vn",
    "vir.com.vn",
    "vietnamplus.vn",
}

REPUTABLE_MEDIA_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "apnews.com",
    "ft.com",
    "wsj.com",
    "cnbc.com",
    "techcrunch.com",
    "dealstreetasia.com",
    "techinasia.com",
}

ENTITY_SEED_SOURCES: dict[str, tuple[dict[str, str], ...]] = {
    "vng": (
        {
            "title": "VNG 2024 annual report - digital business",
            "url": "https://bctn2024.vng.com.vn/about/digital-business",
            "content": "VNG 2024 annual report digital business, go global strategy and AI infrastructure.",
        },
        {
            "title": "VNG 2024 annual report - people strategy",
            "url": "https://bctn2024.vng.com.vn/about/people-strategy",
            "content": "VNG 2024 annual report people strategy, universities and candidate engagement.",
        },
    ),
    "fpt": (
        {
            "title": "FPT strategic directions 2024-2026",
            "url": "https://fptsoftware.com/newsroom/news-and-press-releases/press-release/fpt-unveils-strategic-directions-all-in-on-ai-automotive-and-semiconductor",
            "content": "FPT announced 2024-2026 strategic directions including Artificial Intelligence, Automotive, Semiconductor, Digital Transformation, and Green Transformation.",
        },
        {
            "title": "FPT generative AI solutions at AI4VN 2024",
            "url": "https://fpt.com/en/news/fpt-news/fpt-gay-dau-an-voi-mang-luoi-giai-phap-ai-tao-sinh-tai-ai4vn-2024",
            "content": "FPT presented generative AI solutions at AI4VN 2024.",
        },
        {
            "title": "FPT annual report 2024",
            "url": "https://fpt.com/-/media/project/fpt-corporation/fpt/ir/information-disclosures/year-report/2025/april/20250402---fpt---annual-report-2024.pdf",
            "content": "FPT annual report 2024 with business, partnership, investment, AI, and financial sections.",
        },
        {
            "title": "FPT 20% profit growth after AI, semiconductor, automotive, digital and green strategy",
            "url": "https://fpt.com/en/news/fpt-news/fpt-achieves-20-profit-growth-after-announcement-of-the-ai-semiconductor-automotive-digital-green-strategy-in-2024",
            "content": "FPT reports 20%+ profit growth after announcement of AI, semiconductor, automotive, digital and green strategy in 2024.",
        },
    ),
}


def domain_from_url(url: str) -> str:
    return urlparse(url or "").netloc.lower().removeprefix("www.")


def years_in_text(text: str) -> set[str]:
    return set(_YEAR_RE.findall(text or ""))


def domain_matches(domain: str, candidates: set[str] | tuple[str, ...]) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in candidates)


def entity_from_domain(domain: str) -> str | None:
    for entity, domains in PREFERRED_ENTITY_DOMAINS.items():
        if domain_matches(domain, domains):
            return entity
    return None


def entity_domains_for_query(query: str) -> set[str]:
    domains: set[str] = set()
    for entity in entities_in_text(query):
        domains.update(PREFERRED_ENTITY_DOMAINS.get(entity, ()))
    return domains


def _is_unrelated_pdf(domain: str, url: str, title: str, snippet: str, query: str) -> bool:
    if not urlparse(url or "").path.lower().endswith(".pdf"):
        return False
    if domain_matches(domain, entity_domains_for_query(query)):
        return False
    if domain_matches(domain, PREFERRED_MEDIA_DOMAINS | REPUTABLE_MEDIA_DOMAINS):
        return False
    return bool(entities_in_text(query))


def year_status(query: str, url: str, title: str = "", snippet: str = "") -> YearStatus:
    """Return whether a source has acceptable year evidence for a year-specific query."""
    query_years = years_in_text(query)
    if not query_years:
        return "matched"

    title_snippet_years = years_in_text(f"{title} {snippet}")
    if query_years & title_snippet_years:
        return "matched"

    all_years = years_in_text(f"{title} {snippet} {url}")
    if not all_years:
        return "unknown"
    if query_years & all_years:
        return "matched"
    return "mismatch"


def classify_policy_tier(url: str, title: str = "", snippet: str = "", query: str = "") -> SourcePolicyTier:
    domain = domain_from_url(url)
    if not domain:
        return "blocked"
    if domain_matches(domain, BLOCKED_DOMAINS):
        return "blocked"
    if _is_unrelated_pdf(domain, url, title, snippet, query):
        return "blocked"
    if year_status(query, url, title, snippet) == "mismatch":
        return "blocked"
    if domain_matches(domain, entity_domains_for_query(query)):
        return "preferred"
    if domain_matches(domain, PREFERRED_MEDIA_DOMAINS):
        return "preferred"
    return "allowed"


def seed_source_results(user_query: str, task_question: str) -> list[dict[str, str]]:
    """Return deterministic preferred source seeds for known entities in a task."""
    task_entities = entities_in_text(task_question) or entities_in_text(user_query)
    seeds: list[dict[str, str]] = []
    for entity in sorted(task_entities):
        seeds.extend(ENTITY_SEED_SOURCES.get(entity, ()))
    return list({seed["url"]: seed for seed in seeds}.values())
