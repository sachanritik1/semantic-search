from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from app.config import settings
from app.adapters.llm.base import BaseLLM, LLMResponse


def _build_input(prompt: str, system_prompt: str | None) -> str | list[dict[str, str]]:
    if not system_prompt:
        return prompt
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _extract_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    result: dict[str, Any] = {}
    if getattr(usage, "input_tokens", None) is not None:
        result["input_tokens"] = usage.input_tokens
    if getattr(usage, "output_tokens", None) is not None:
        result["output_tokens"] = usage.output_tokens
    if getattr(usage, "total_tokens", None) is not None:
        result["total_tokens"] = usage.total_tokens

    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is not None:
            result["cached_tokens"] = cached

    return result or None


class OpenaiLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = OpenAI(api_key=api_key)

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
        use_model = model or self.model
        kwargs: dict[str, Any] = {
            "model": use_model,
            "input": _build_input(prompt, system_prompt),
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if settings.PROMPT_CACHE_ENABLED and cache_key:
            kwargs["prompt_cache_key"] = cache_key

        response = self.client.responses.create(**kwargs)

        if not response or not response.output_text:
            raise ValueError("No text returned from OpenAI API")

        return LLMResponse(
            content=response.output_text,
            model=use_model,
            raw_response=response,
            usage=_extract_usage(response),
        )

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
        use_model = model or self.model
        kwargs: dict[str, Any] = {
            "model": use_model,
            "input": _build_input(prompt, system_prompt),
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if settings.PROMPT_CACHE_ENABLED and cache_key:
            kwargs["prompt_cache_key"] = cache_key

        with self.client.responses.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield delta
