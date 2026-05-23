# app/llm/gemini_llm.py

from collections.abc import Iterator
from typing import Any

from google import genai
from google.genai import types

from app.llm.base import BaseLLM, LLMResponse


def _build_config(
    *,
    temperature: float,
    max_tokens: int | None,
    system_prompt: str | None,
) -> types.GenerateContentConfig:
    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if system_prompt:
        kwargs["system_instruction"] = system_prompt
    return types.GenerateContentConfig(**kwargs)


def _extract_usage(response: Any) -> dict[str, Any] | None:
    if not hasattr(response, "usage_metadata") or not response.usage_metadata:
        return None

    meta = response.usage_metadata
    usage: dict[str, Any] = {
        "prompt_tokens": meta.prompt_token_count,
        "completion_tokens": meta.candidates_token_count,
        "total_tokens": meta.total_token_count,
    }
    cached = getattr(meta, "cached_content_token_count", None)
    if cached is not None:
        usage["cached_content_token_count"] = cached
    return usage


class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = genai.Client(api_key=api_key)

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
        del cache_key  # Gemini implicit cache uses system_instruction; no cache_key API yet
        use_model = model or self.model

        response = self.client.models.generate_content(  # type: ignore
            model=use_model,
            contents=prompt,
            config=_build_config(
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            ),
        )

        if not response or response.text is None:
            raise ValueError("No text returned from Gemini API")

        return LLMResponse(
            content=response.text,
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
        del cache_key
        use_model = model or self.model

        for chunk in self.client.models.generate_content_stream(  # type: ignore
            model=use_model,
            contents=prompt,
            config=_build_config(
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            ),
        ):
            if chunk.text:
                yield chunk.text
