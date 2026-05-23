# app/utils/prompt_cache.py

from app.config import settings


def cache_key(name: str) -> str | None:
    """Build a versioned provider cache key, or None when caching is disabled."""
    if not settings.PROMPT_CACHE_ENABLED:
        return None
    return f"{name}:{settings.PROMPT_CACHE_VERSION}"
