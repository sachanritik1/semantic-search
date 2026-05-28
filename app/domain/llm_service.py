# app/services/llm_service.py

from collections.abc import AsyncIterator

from app.adapters.llm.base import BaseLLM, LLMResponse


class LLMService:
    """Thin application wrapper around a decorated BaseLLM.

    Keeps the existing ``generate_text`` / ``generate_text_async`` /
    ``stream_text`` interface so callers do not need to change.
    All cross-cutting concerns (cost tracking, usage logging, Langfuse
    tracing) have been moved into composable decorator adapters.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> LLMResponse:
        return self.llm.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        )

    async def generate_text_async(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> LLMResponse:
        return await self.llm.generate_async(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        )

    async def stream_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in self.llm.stream_generate_async(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        ):
            yield chunk
