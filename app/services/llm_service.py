# app/services/llm_service.py

from collections.abc import AsyncIterator

from langfuse import get_client, observe

from app.llm.base import BaseLLM, LLMResponse
from app.utils.langfuse_usage import to_langfuse_usage
from app.utils.llm_usage import annotate_cost, log_llm_usage


class LLMService:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def _finalize(
        self,
        response: LLMResponse,
        *,
        prompt: str,
        system_prompt: str | None,
        context: str,
    ) -> LLMResponse:
        annotate_cost(response.usage, model=response.model)
        log_llm_usage(response.usage, context=context, model=response.model)
        generation_input: str | dict[str, str] = (
            {"prompt": prompt, "system_prompt": system_prompt}
            if system_prompt
            else prompt
        )
        get_client().update_current_generation(
            input=generation_input,
            output=response.content,
            model=response.model,
            usage_details=to_langfuse_usage(response.usage),
            metadata={"context": context},
        )
        return response

    @observe(name="llm.generate_text", as_type="generation", capture_input=False)
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
        """
        Application-level LLM call.
        """
        response = self.llm.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        )
        return self._finalize(
            response,
            prompt=prompt,
            system_prompt=system_prompt,
            context="generate",
        )

    @observe(name="llm.generate_text_async", as_type="generation", capture_input=False)
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
        response = await self.llm.generate_async(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        )
        return self._finalize(
            response,
            prompt=prompt,
            system_prompt=system_prompt,
            context="generate_async",
        )

    @observe(name="llm.stream_text", as_type="generation", capture_input=False)
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
        use_model = model or self.llm.model
        get_client().update_current_generation(
            input=prompt,
            model=use_model,
            metadata={"system_prompt": system_prompt, "context": "stream"},
        )
        async for chunk in self.llm.stream_generate_async(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
            cache_key=cache_key,
        ):
            yield chunk
