"""Cheap, deterministic checks on a final report."""
from __future__ import annotations

import re
from urllib.parse import urlparse

CitationCheck = tuple[bool, str]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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


def check_claims_supported(state: dict, min_overlap: float = 0.15) -> CitationCheck:
    """Offline faithfulness proxy: each claim's statement should share tokens
    with its snippet. Pure-Python; no LLM call required."""
    overlaps: list[float] = []
    for f in state.get("findings", []) or []:
        for c in (f.get("claims") or []):
            stmt = _tokens(c.get("statement", ""))
            snip = _tokens(c.get("snippet", ""))
            overlaps.append(_jaccard(stmt, snip))
    if not overlaps:
        return (False, "no structured claims to check")
    avg = sum(overlaps) / len(overlaps)
    return (avg >= min_overlap, f"avg jaccard(statement, snippet)={avg:.2f} (≥{min_overlap:.2f})")


def check_dead_links(report: str, timeout: float = 5.0, max_dead: int = 1) -> CitationCheck:
    """HEAD every URL in the report; ≤max_dead allowed to fail."""
    urls = re.findall(r"https?://\S+", report)
    urls = [u.rstrip(".,);:") for u in urls]
    urls = list(dict.fromkeys(urls))  # dedupe preserving order
    if not urls:
        return (False, "no URLs to check")
    try:
        import requests  # local import: keep heuristics light when offline
    except ImportError:
        return (False, "requests not available")
    dead: list[str] = []
    for u in urls:
        try:
            r = requests.head(u, timeout=timeout, allow_redirects=True)
            if r.status_code >= 400:
                # Some servers reject HEAD; retry with GET (small range).
                r = requests.get(u, timeout=timeout, stream=True)
                if r.status_code >= 400:
                    dead.append(u)
                r.close()
        except Exception:
            dead.append(u)
    return (len(dead) <= max_dead, f"{len(dead)}/{len(urls)} dead links (max={max_dead})")


CHECKS = [
    ("citations", check_report_has_citations),
    ("sources_section", check_sources_section),
    ("urls_valid", check_urls_valid),
    ("no_empty_sections", check_no_empty_sections),
    ("length_reasonable", check_length_reasonable),
]


def run_heuristic_checks(state: dict, *, check_links: bool = False) -> dict:
    """Run all cheap checks. Set check_links=True to enable the network-bound
    dead-link check (off by default to keep eval runs offline-friendly)."""
    report = state.get("final_report") or state.get("draft_report") or ""
    results: dict[str, dict] = {}
    for name, fn in CHECKS:
        ok, msg = fn(report)
        results[name] = {"passed": ok, "message": msg}
    ok, msg = check_iterations_terminated(state)
    results["iterations_terminated"] = {"passed": ok, "message": msg}
    ok, msg = check_claims_supported(state)
    results["claims_supported"] = {"passed": ok, "message": msg}
    if check_links:
        ok, msg = check_dead_links(report)
        results["dead_links"] = {"passed": ok, "message": msg}
    passed = sum(1 for r in results.values() if r["passed"])
    results["_summary"] = {
        "pass_rate": passed / len(results) if results else 0.0,
        "passed": passed,
        "total": len(results),
    }
    return results
