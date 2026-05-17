"""Small provider key/model rotation helpers for development-stage quotas."""
from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from app.logger import log_event

T = TypeVar("T")

_LIMIT_RE = re.compile(
    r"(rate limit|too many requests|429|quota|resource_exhausted|tokens per minute|"
    r"requests per minute|tpm|rpm|rpd|tpd|insufficient_quota|503|unavailable|"
    r"service unavailable|temporarily unavailable|high demand|overloaded|413|"
    r"request too large|rate_limit_exceeded)",
    re.IGNORECASE,
)


def split_env_list(name: str, fallback_name: str | None = None, default: str = "") -> list[str]:
    raw = os.getenv(name)
    if not raw and fallback_name:
        raw = os.getenv(fallback_name)
    if not raw:
        raw = default
    return [part.strip() for part in raw.split(",") if part.strip()]


def is_limit_error(exc: Exception) -> bool:
    return bool(_LIMIT_RE.search(f"{type(exc).__name__}: {exc}"))


def retry_after_seconds(exc: Exception, default: float = 1.0) -> float:
    text = str(exc)
    match = re.search(r"retry[- ]after[:= ]+(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return min(30.0, max(0.0, float(match.group(1))))
    match = re.search(
        r"(?:try again|retry) in (\d+(?:\.\d+)?)\s*"
        r"(ms|s|sec|secs|second|seconds|m|min|minute|minutes)",
        text,
        re.IGNORECASE,
    )
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "ms":
            value /= 1000
        elif unit.startswith("m"):
            value *= 60
        return min(30.0, max(0.0, value))
    return default


class Ring:
    """Thread-safe round-robin over a small list of strings."""

    def __init__(self, name: str, values: list[str]):
        self.name = name
        self.values = values or [""]
        self.index = 0
        self._lock = threading.Lock()

    def current(self) -> str:
        with self._lock:
            return self.values[self.index % len(self.values)]

    def rotate(self) -> str:
        with self._lock:
            self.index = (self.index + 1) % len(self.values)
            return self.values[self.index]

    def has_multiple(self) -> bool:
        return len([v for v in self.values if v]) > 1


GROQ_KEYS = Ring("groq_key", split_env_list("GROQ_API_KEYS", "GROQ_API_KEY"))
GOOGLE_KEYS = Ring("google_key", split_env_list("GOOGLE_API_KEYS", "GOOGLE_API_KEY"))
TAVILY_KEYS = Ring("tavily_key", split_env_list("TAVILY_API_KEYS", "TAVILY_API_KEY"))

GROQ_PLANNER_MODELS = Ring(
    "groq_planner_model",
    split_env_list("GROQ_PLANNER_MODELS", default="llama-3.3-70b-versatile,llama-3.1-8b-instant"),
)
GROQ_RESEARCHER_MODELS = Ring(
    "groq_researcher_model",
    split_env_list("GROQ_RESEARCHER_MODELS", default="llama-3.1-8b-instant,llama-3.3-70b-versatile"),
)
GROQ_EXTRACTOR_MODELS = Ring(
    "groq_extractor_model",
    split_env_list("GROQ_EXTRACTOR_MODELS", default="llama-3.3-70b-versatile,llama-3.1-8b-instant"),
)
GROQ_SYNTHESIZER_MODELS = Ring(
    "groq_synthesizer_model",
    split_env_list("GROQ_SYNTHESIZER_MODELS", default="llama-3.3-70b-versatile,llama-3.1-8b-instant"),
)
GOOGLE_CRITIC_MODELS = Ring(
    "google_critic_model",
    split_env_list("GOOGLE_CRITIC_MODELS", default="gemini-2.5-flash,gemini-2.5-flash-lite"),
)


def rotate_groq(reason: str = "") -> None:
    key = GROQ_KEYS.rotate()
    model_values = [
        GROQ_PLANNER_MODELS.rotate(),
        GROQ_RESEARCHER_MODELS.rotate(),
        GROQ_EXTRACTOR_MODELS.rotate(),
        GROQ_SYNTHESIZER_MODELS.rotate(),
    ]
    log_event(
        "provider_rotation",
        "-",
        provider="groq",
        reason=reason,
        has_key=bool(key),
        models=model_values,
    )


def rotate_google(reason: str = "") -> None:
    key = GOOGLE_KEYS.rotate()
    model = GOOGLE_CRITIC_MODELS.rotate()
    log_event(
        "provider_rotation",
        "-",
        provider="google",
        reason=reason,
        has_key=bool(key),
        model=model,
    )


def rotate_tavily(reason: str = "") -> None:
    key = TAVILY_KEYS.rotate()
    log_event("provider_rotation", "-", provider="tavily", reason=reason, has_key=bool(key))


def invoke_with_rotation(
    provider: str,
    call: Callable[[], T],
    *,
    attempts: int = 2,
    rotate: Callable[[str], None] | None = None,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return call()
        except Exception as exc:
            last_exc = exc
            if not is_limit_error(exc) or attempt >= attempts - 1:
                raise
            if rotate:
                rotate(f"{type(exc).__name__}: {exc}")
            sleep_s = retry_after_seconds(exc)
            log_event(
                "provider_retry",
                "-",
                provider=provider,
                attempt=attempt + 1,
                sleep_seconds=sleep_s,
                error=f"{type(exc).__name__}: {exc}",
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc
