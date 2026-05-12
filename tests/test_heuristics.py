"""Heuristic check unit tests."""
from app.evaluation.heuristics import (
    check_claims_supported,
    check_length_reasonable,
    check_no_empty_sections,
    check_report_has_citations,
    check_sources_section,
    check_urls_valid,
    run_heuristic_checks,
)

_LOREM = (
    "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute "
    "irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat "
    "nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui "
    "officia deserunt mollit anim id est laborum "
)
GOOD_REPORT = (
    "# Title\n\n"
    "## Executive Summary\n"
    "This is the summary [1].\n\n"
    "## 1. Section\n"
    "Some claim with citation [1]. Another claim [2]. Third claim [3]. "
    + (_LOREM * 6)
    + "\n\n## Sources\n[1] https://example.com/a\n[2] https://example.com/b\n"
    "[3] https://example.com/c\n"
)


def test_good_report_passes_all():
    state = {
        "final_report": GOOD_REPORT,
        "current_iteration": 2,
        "findings": [
            {
                "claims": [
                    {
                        "statement": "Example company reported strong revenue",
                        "snippet": "The example company reported strong revenue this quarter",
                    }
                ]
            }
        ],
    }
    res = run_heuristic_checks(state)
    assert res["citations"]["passed"]
    assert res["sources_section"]["passed"]
    assert res["urls_valid"]["passed"]
    assert res["length_reasonable"]["passed"]
    assert res["iterations_terminated"]["passed"]
    assert res["claims_supported"]["passed"]
    assert res["_summary"]["pass_rate"] >= 0.8


def test_short_report_fails_length():
    ok, _ = check_length_reasonable("too short")
    assert not ok


def test_no_citations_fails():
    ok, _ = check_report_has_citations("plain text with no citations")
    assert not ok


def test_no_sources_section_fails():
    ok, _ = check_sources_section("# Title\nblah")
    assert not ok


def test_invalid_urls_caught():
    # URL with no host (just scheme + path) should fail validation.
    ok, _ = check_urls_valid("see http:///path-only-no-host")
    assert not ok


def test_empty_sections_caught():
    bad = "# T\n\n## X\n\n## Y\n\n## Z\nactual content here"
    ok, _ = check_no_empty_sections(bad)
    assert not ok


def test_claims_supported_passes_when_snippet_overlaps():
    state = {
        "findings": [
            {
                "claims": [
                    {
                        "statement": "VNG reported Q3 2024 revenue of 2 trillion VND",
                        "snippet": "VNG announced Q3 2024 revenue at 2 trillion VND in its filing",
                    },
                ]
            }
        ]
    }
    ok, _ = check_claims_supported(state)
    assert ok


def test_claims_supported_fails_when_no_overlap():
    state = {
        "findings": [
            {
                "claims": [
                    {
                        "statement": "Tigers walk on the moon every Tuesday",
                        "snippet": "Soccer rules state offside positions cannot score",
                    },
                ]
            }
        ]
    }
    ok, _ = check_claims_supported(state)
    assert not ok


def test_claims_supported_no_claims_fails():
    ok, _ = check_claims_supported({"findings": []})
    assert not ok
