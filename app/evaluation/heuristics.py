"""Cheap, deterministic checks on a final report."""
from __future__ import annotations

import re
from urllib.parse import urlparse

CitationCheck = tuple[bool, str]


def check_report_has_citations(report: str, min_count: int = 3) -> CitationCheck:
    found = re.findall(r"\[\d+\]", report)
    n = len({c for c in found})
    return (n >= min_count, f"found {n} unique [N] citations (min={min_count})")


def check_sources_section(report: str) -> CitationCheck:
    has = bool(re.search(r"(?im)^\#{1,3}\s+Sources\b", report))
    return (has, "has '## Sources' section" if has else "missing '## Sources' section")


def check_urls_valid(report: str) -> CitationCheck:
    urls = re.findall(r"https?://\S+", report)
    bad = [u for u in urls if not urlparse(u).netloc]
    if not urls:
        return (False, "no URLs found")
    return (not bad, f"{len(urls)} URLs, {len(bad)} invalid")


def check_no_empty_sections(report: str) -> CitationCheck:
    sections = re.split(r"(?m)^\#{1,3}\s+.+$", report)
    short = [i for i, s in enumerate(sections[1:], start=1) if len(s.strip()) < 50]
    return (not short, f"{len(short)} short/empty sections")


def check_length_reasonable(report: str, lo: int = 300, hi: int = 3000) -> CitationCheck:
    n = len(report.split())
    return (lo <= n <= hi, f"{n} words (target {lo}-{hi})")


def check_iterations_terminated(state: dict, max_iter: int = 3) -> CitationCheck:
    n = state.get("current_iteration", 0)
    return (n <= max_iter, f"iterations={n} (max={max_iter})")


CHECKS = [
    ("citations", check_report_has_citations),
    ("sources_section", check_sources_section),
    ("urls_valid", check_urls_valid),
    ("no_empty_sections", check_no_empty_sections),
    ("length_reasonable", check_length_reasonable),
]


def run_heuristic_checks(state: dict) -> dict:
    report = state.get("final_report") or state.get("draft_report") or ""
    results: dict[str, dict] = {}
    for name, fn in CHECKS:
        ok, msg = fn(report)
        results[name] = {"passed": ok, "message": msg}
    ok, msg = check_iterations_terminated(state)
    results["iterations_terminated"] = {"passed": ok, "message": msg}
    passed = sum(1 for r in results.values() if r["passed"])
    results["_summary"] = {
        "pass_rate": passed / len(results) if results else 0.0,
        "passed": passed,
        "total": len(results) - 0,
    }
    return results
