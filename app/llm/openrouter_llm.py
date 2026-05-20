# app/llm/openrouter_llm.py

from collections.abc import Iterator

from langchain_openai import ChatOpenAI

from app.llm.base import BaseLLM, LLMResponse
from app.utils.llm_content import normalize_llm_content


class OpenRouterLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        if not model:
            raise ValueError("OPENROUTER_MODEL is required when LLM_PROVIDER=openrouter")

        self.model = model
        # OpenRouter exposes an OpenAI-compatible API under this base URL.
        self.client = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        use_model = model or self.model

        bound = self.client.bind(
            model=use_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = bound.invoke(prompt)
        content = response.content
        if not content:
            content = ""

        usage = None
        metadata = getattr(response, "response_metadata", None)
        if metadata and "token_usage" in metadata:
            usage = metadata["token_usage"]

        return LLMResponse(
            content=normalize_llm_content(content),
            model=use_model,
            raw_response=response,
            usage=usage,
        )

    def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        use_model = model or self.model

        bound = self.client.bind(
            model=use_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for chunk in bound.stream(prompt):
            text = _chunk_to_text(chunk.content)
            if text:
                yield text


def _chunk_to_text(content) -> str:
    """Extract a streaming delta's text WITHOUT stripping whitespace.

    Per-chunk normalization must preserve leading/trailing spaces between
    tokens; `normalize_llm_content` is for full responses and strips them.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text is None:
                    text = block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)
