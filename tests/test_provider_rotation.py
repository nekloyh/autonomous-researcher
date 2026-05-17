"""Tests for provider key/model rotation helpers."""
from __future__ import annotations

import pytest

from app.provider_rotation import Ring, invoke_with_rotation, is_limit_error, retry_after_seconds


def test_ring_rotates_values():
    ring = Ring("x", ["a", "b"])
    assert ring.current() == "a"
    assert ring.rotate() == "b"
    assert ring.rotate() == "a"


def test_is_limit_error_matches_common_quota_messages():
    assert is_limit_error(RuntimeError("429 rate limit exceeded"))
    assert is_limit_error(RuntimeError("RESOURCE_EXHAUSTED: quota exceeded"))
    assert is_limit_error(RuntimeError("503 UNAVAILABLE: model is experiencing high demand"))
    assert is_limit_error(RuntimeError("413 request too large: rate_limit_exceeded"))
    assert not is_limit_error(RuntimeError("schema parse failed"))


def test_invoke_with_rotation_retries_limit_once():
    calls = {"n": 0, "rotated": 0}

    def call():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 rate limit exceeded retry-after: 0")
        return "ok"

    out = invoke_with_rotation(
        "test",
        call,
        attempts=2,
        rotate=lambda reason: calls.__setitem__("rotated", calls["rotated"] + 1),
    )
    assert out == "ok"
    assert calls == {"n": 2, "rotated": 1}


def test_retry_after_seconds_parses_groq_try_again_message():
    assert retry_after_seconds(RuntimeError("Please try again in 4.95s.")) == 4.95
    assert retry_after_seconds(RuntimeError("Please try again in 5m13.632s.")) == 30.0
    assert retry_after_seconds(RuntimeError("Please retry in 27.102080279s.")) == 27.102080279


def test_invoke_with_rotation_does_not_retry_non_limit_errors():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise RuntimeError("parse failed")

    with pytest.raises(RuntimeError, match="parse failed"):
        invoke_with_rotation("test", call, attempts=2)
    assert calls["n"] == 1
