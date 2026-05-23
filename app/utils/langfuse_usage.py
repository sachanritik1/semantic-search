# app/utils/langfuse_usage.py

from typing import Any


def to_langfuse_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map provider-specific token usage to Langfuse usage_details keys."""
    if not usage:
        return None
    out: dict[str, Any] = {}
    if (v := usage.get("input_tokens") or usage.get("prompt_tokens")) is not None:
        out["input"] = v
    if (v := usage.get("output_tokens") or usage.get("completion_tokens")) is not None:
        out["output"] = v
    if (v := usage.get("total_tokens")) is not None:
        out["total"] = v
    if (v := usage.get("cached_tokens") or usage.get("cached_content_token_count")) is not None:
        out["cache_read_input_tokens"] = v
    return out or None
