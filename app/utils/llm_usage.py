# app/utils/llm_usage.py

import logging
from typing import Any

from app.utils.llm_pricing import estimate_cost

logger = logging.getLogger(__name__)


def _extract_cached_tokens(usage: dict[str, Any]) -> int | None:
    return (
        usage.get("cached_tokens")
        or usage.get("cached_content_token_count")
        or usage.get("cache_read_input_tokens")
    )


def annotate_cost(
    usage: dict[str, Any] | None, *, model: str | None
) -> dict[str, Any] | None:
    """Compute estimated cost and attach it to ``usage`` in-place.

    OpenRouter (and some providers) include a numeric ``cost`` field on usage.
    We store our breakdown under ``estimated_cost`` and only set ``cost`` when
    it is not already a provider-reported number.

    Returns the (mutated) usage dict for chaining; returns ``None`` when usage
    itself is ``None``.
    """
    if usage is None:
        return None
    estimated = estimate_cost(model, usage)
    if estimated is not None:
        usage["estimated_cost"] = estimated
        existing = usage.get("cost")
        if not isinstance(existing, (int, float)):
            usage["cost"] = estimated
    return usage


def log_llm_usage(
    usage: dict[str, Any] | None,
    *,
    context: str,
    model: str | None = None,
) -> None:
    """Log prompt-cache token counts and estimated cost when available."""
    if not usage:
        return

    cached = _extract_cached_tokens(usage)
    created = usage.get("cache_creation_input_tokens")
    if cached is not None:
        if created is not None:
            logger.info(
                "Prompt cache %s model=%s: cached_tokens=%s cache_creation_input_tokens=%s",
                context,
                model,
                cached,
                created,
            )
        else:
            logger.info(
                "Prompt cache %s model=%s: cached_tokens=%s",
                context,
                model,
                cached,
            )

    estimated = usage.get("estimated_cost")
    if isinstance(estimated, dict):
        logger.info(
            "LLM cost %s model=%s: total=$%.6f input=$%.6f cached=$%.6f output=$%.6f savings=$%.6f",
            context,
            model,
            estimated.get("total_cost", 0.0),
            estimated.get("input_cost", 0.0),
            estimated.get("cached_input_cost", 0.0),
            estimated.get("output_cost", 0.0),
            estimated.get("savings_from_cache", 0.0),
        )
        return

    cost = usage.get("cost")
    if isinstance(cost, dict):
        logger.info(
            "LLM cost %s model=%s: total=$%.6f input=$%.6f cached=$%.6f output=$%.6f savings=$%.6f",
            context,
            model,
            cost.get("total_cost", 0.0),
            cost.get("input_cost", 0.0),
            cost.get("cached_input_cost", 0.0),
            cost.get("output_cost", 0.0),
            cost.get("savings_from_cache", 0.0),
        )
    elif isinstance(cost, (int, float)):
        logger.info(
            "LLM cost %s model=%s: total=$%.6f (provider-reported)",
            context,
            model,
            float(cost),
        )


def log_prompt_cache_usage(usage: dict[str, Any] | None, *, context: str) -> None:
    """Backward-compatible wrapper; prefer ``log_llm_usage``."""
    log_llm_usage(usage, context=context)
