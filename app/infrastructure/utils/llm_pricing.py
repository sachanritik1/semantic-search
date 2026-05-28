# app/utils/llm_pricing.py

"""Per-model LLM pricing and cost estimation.

Prices are stored in **USD per 1M tokens**. Cached input tokens are billed at
the provider's reduced rate when the model supports prompt caching; tokens
that have to be (re)written to the cache are billed at `cache_write` when
provided (Anthropic-style ephemeral cache) and otherwise at `input`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# USD per 1,000,000 tokens. Keep entries lowercase; lookups are case-insensitive.
# Source: provider docs as of mid-2026. Update via `register_pricing()` or the
# `LLM_PRICING_JSON_PATH` env var for new models without code changes.
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
    # Google Gemini
    "gemini-1.5-flash": {"input": 0.075, "cached_input": 0.01875, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "cached_input": 0.3125, "output": 5.00},
    "gemini-2.5-flash": {"input": 0.30, "cached_input": 0.075, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "cached_input": 0.31, "output": 10.00},
    # Anthropic (direct + via OpenRouter)
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "cached_input": 0.30,
        "cache_write": 3.75,
        "output": 15.00,
    },
    "anthropic/claude-3.5-sonnet": {
        "input": 3.00,
        "cached_input": 0.30,
        "cache_write": 3.75,
        "output": 15.00,
    },
    "anthropic/claude-sonnet-4": {
        "input": 3.00,
        "cached_input": 0.30,
        "cache_write": 3.75,
        "output": 15.00,
    },
    "anthropic/claude-haiku-4": {
        "input": 0.80,
        "cached_input": 0.08,
        "cache_write": 1.00,
        "output": 4.00,
    },
    # OpenRouter passthroughs
    "openai/gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "openai/gpt-oss-120b:free": {"input": 0.0, "cached_input": 0.0, "output": 0.0},
    "google/gemini-2.5-flash": {"input": 0.30, "cached_input": 0.075, "output": 2.50},
}

_PRICING: dict[str, dict[str, float]] = {
    k.lower(): v for k, v in _DEFAULT_PRICING.items()
}
_OVERRIDES_LOADED = False


def register_pricing(
    model: str,
    *,
    input: float,
    output: float,
    cached_input: float | None = None,
    cache_write: float | None = None,
) -> None:
    """Add or override pricing for a model (USD per 1M tokens)."""
    rates: dict[str, float] = {"input": input, "output": output}
    if cached_input is not None:
        rates["cached_input"] = cached_input
    if cache_write is not None:
        rates["cache_write"] = cache_write
    _PRICING[model.lower()] = rates


def _load_overrides_from_env() -> None:
    global _OVERRIDES_LOADED
    if _OVERRIDES_LOADED:
        return
    _OVERRIDES_LOADED = True

    path = os.environ.get("LLM_PRICING_JSON_PATH")
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load LLM pricing overrides from %s: %s", path, exc)
        return

    if not isinstance(data, dict):
        logger.warning("LLM pricing override at %s must be a JSON object", path)
        return

    for model, rates in data.items():
        if not isinstance(rates, dict) or "input" not in rates or "output" not in rates:
            logger.warning("Skipping malformed pricing entry for %s", model)
            continue
        register_pricing(
            model,
            input=float(rates["input"]),
            output=float(rates["output"]),
            cached_input=float(rates["cached_input"])
            if "cached_input" in rates
            else None,
            cache_write=float(rates["cache_write"]) if "cache_write" in rates else None,
        )


def _lookup_rates(model: str) -> dict[str, float] | None:
    key = model.lower()
    rates = _PRICING.get(key)
    if rates is not None:
        return rates
    # Try unqualified suffix (e.g. "openai/gpt-4o" -> "gpt-4o")
    if "/" in key:
        rates = _PRICING.get(key.rsplit("/", 1)[1])
        if rates is not None:
            return rates
    # Try stripping trailing tags (e.g. "gpt-4o-2024-11-20" -> "gpt-4o")
    parts = key.split("-")
    while len(parts) > 1:
        parts.pop()
        candidate = "-".join(parts)
        if candidate in _PRICING:
            return _PRICING[candidate]
    return None


def _normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    cached = (
        usage.get("cached_tokens")
        or usage.get("cached_content_token_count")
        or usage.get("cache_read_input_tokens")
        or 0
    )
    cache_write = usage.get("cache_creation_input_tokens") or 0
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "cached_tokens": int(cached),
        "cache_write_tokens": int(cache_write),
    }


def estimate_cost(
    model: str | None,
    usage: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Estimate USD cost from a normalized or raw usage dict.

    Returns ``None`` when the model is unknown or usage is missing; callers
    should treat this as "cost unavailable" rather than zero.
    """
    if not model or not usage:
        return None

    _load_overrides_from_env()
    rates = _lookup_rates(model)
    if rates is None:
        logger.debug(
            "No pricing registered for model=%s; skipping cost estimate", model
        )
        return None

    n = _normalize_usage(usage)
    cached = n["cached_tokens"]
    fresh_input = max(n["prompt_tokens"] - cached, 0)
    cache_write = n["cache_write_tokens"]

    per_token = 1_000_000.0
    input_cost = fresh_input * rates["input"] / per_token
    cached_rate = rates.get("cached_input", rates["input"])
    cached_cost = cached * cached_rate / per_token
    write_rate = rates.get("cache_write", rates["input"])
    cache_write_cost = cache_write * write_rate / per_token
    output_cost = n["completion_tokens"] * rates["output"] / per_token

    total = input_cost + cached_cost + cache_write_cost + output_cost
    if cached and rates["input"] > 0:
        saved = cached * (rates["input"] - cached_rate) / per_token
    else:
        saved = 0.0

    return {
        "currency": "USD",
        "input_cost": round(input_cost, 8),
        "cached_input_cost": round(cached_cost, 8),
        "cache_write_cost": round(cache_write_cost, 8),
        "output_cost": round(output_cost, 8),
        "total_cost": round(total, 8),
        "savings_from_cache": round(saved, 8),
    }
