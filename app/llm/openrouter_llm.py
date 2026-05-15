# app/llm/openrouter_llm.py

from openai import OpenAI
from app.llm.base import BaseLLM, LLMResponse


class OpenRouterLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        if not model:
            raise ValueError("OPENROUTER_MODEL is required when LLM_PROVIDER=openrouter")

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            # OpenRouter exposes an OpenAI-compatible API under this base URL.
            base_url="https://openrouter.ai/api/v1",
        )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not response or not response.choices:
            raise ValueError("No response from OpenRouter API")

        content = response.choices[0].message.content
        if not content:
            raise ValueError("No text content in OpenRouter response")

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            model=self.model,
            raw_response=response,
            usage=usage,
        )
