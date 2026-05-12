"""Rough USD cost estimator. Provider rates change; treat as a sanity-check
order-of-magnitude estimate, not an invoice."""
from __future__ import annotations

# Rates are USD per million tokens (input/output averaged, approximate Apr 2026
# public rates). Update as needed.
PROVIDER_RATE_USD_PER_M_TOKEN = {
    "groq:gemma2-9b-it": 0.20,
    "groq:llama-3.1-8b-instant": 0.10,
    "groq:llama-3.3-70b-versatile": 0.69,
    "gemini:gemini-2.0-flash": 0.10,
}

# Conservative blended rate for "unknown model" inputs.
_DEFAULT_RATE = 0.50


def estimate_cost_usd(tokens_by_model: dict[str, int]) -> float:
    total = 0.0
    for model_key, tokens in tokens_by_model.items():
        rate = PROVIDER_RATE_USD_PER_M_TOKEN.get(model_key, _DEFAULT_RATE)
        total += rate * tokens / 1_000_000
    return total


def estimate_cost_simple(total_tokens: int) -> float:
    """When we only have a flat token count, apply the blended default rate."""
    return _DEFAULT_RATE * total_tokens / 1_000_000
