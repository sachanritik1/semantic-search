# app/llm/decorators.py

from collections.abc import AsyncIterator, Iterator

from langfuse import get_client, observe

from app.llm.base import BaseLLM, LLMResponse
from app.utils.langfuse_usage import to_langfuse_usage
from app.utils.llm_usage import annotate_cost, log_llm_usage


class CostTrackingLLM(BaseLLM):
    """
    Decorator that adds cost annotation and usage logging to LLM calls.
    """

    def __init__(self, inner: BaseLLM):
        self._inner = inner

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> LLMResponse:
        response = self._inner.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        )
        annotate_cost(response.usage, model=response.model)
        log_llm_usage(response.usage, context="generate", model=response.model)
        return response

    async def generate_async(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> LLMResponse:
        response = await self._inner.generate_async(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        )
        annotate_cost(response.usage, model=response.model)
        log_llm_usage(response.usage, context="generate_async", model=response.model)
        return response

    def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> Iterator[str]:
        yield from self._inner.stream_generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        )

    async def stream_generate_async(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in self._inner.stream_generate_async(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        ):
            yield chunk


class TracingLLM(BaseLLM):
    """
    Decorator that adds Langfuse tracing and generation tracking to LLM calls.
    """

    def __init__(self, inner: BaseLLM):
        self._inner = inner

    @observe(name="llm.generate", as_type="generation", capture_input=False)
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> LLMResponse:
        generation_input: str | dict[str, str] = (
            {"prompt": prompt, "system_prompt": system_prompt}
            if system_prompt
            else prompt
        )
        response = self._inner.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        )
        get_client().update_current_generation(
            input=generation_input,
            output=response.content,
            model=response.model,
            usage_details=to_langfuse_usage(response.usage),
            metadata={"context": "generate"},
        )
        return response

    @observe(name="llm.generate_async", as_type="generation", capture_input=False)
    async def generate_async(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> LLMResponse:
        generation_input: str | dict[str, str] = (
            {"prompt": prompt, "system_prompt": system_prompt}
            if system_prompt
            else prompt
        )
        response = await self._inner.generate_async(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        )
        get_client().update_current_generation(
            input=generation_input,
            output=response.content,
            model=response.model,
            usage_details=to_langfuse_usage(response.usage),
            metadata={"context": "generate_async"},
        )
        return response

    @observe(name="llm.stream_generate", as_type="generation", capture_input=False)
    def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> Iterator[str]:
        use_model = model or getattr(self._inner, "model", None)
        get_client().update_current_generation(
            input=prompt,
            model=use_model,
            metadata={"system_prompt": system_prompt, "context": "stream_generate"},
        )
        yield from self._inner.stream_generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        )

    @observe(name="llm.stream_generate_async", as_type="generation", capture_input=False)
    async def stream_generate_async(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        cache_key: str | None = None,
    ) -> AsyncIterator[str]:
        use_model = model or getattr(self._inner, "model", None)
        get_client().update_current_generation(
            input=prompt,
            model=use_model,
            metadata={"system_prompt": system_prompt, "context": "stream_generate_async"},
        )
        async for chunk in self._inner.stream_generate_async(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        ):
            yield chunk
