# app/llm/factory.py

from app.config import settings
from app.llm.base import BaseLLM
from app.llm.openai_llm import OpenaiLLM
from app.llm.gemini_llm import GeminiLLM


def get_llm() -> BaseLLM:
    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        return OpenaiLLM(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
        )

    if provider == "gemini":
        return GeminiLLM(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
