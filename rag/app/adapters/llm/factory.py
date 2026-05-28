# app/llm/factory.py

from app.config import settings
from app.adapters.llm.base import BaseLLM
from app.adapters.llm.decorators import CostTrackingLLM, TracingLLM
from app.adapters.llm.openai import OpenaiLLM
from app.adapters.llm.gemini import GeminiLLM
from app.adapters.llm.openrouter import OpenRouterLLM


def get_llm() -> BaseLLM:
    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        raw = OpenaiLLM(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
        )
    elif provider == "gemini":
        raw = GeminiLLM(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
        )
    elif provider == "openrouter":
        raw = OpenRouterLLM(
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_MODEL,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    stack = CostTrackingLLM(raw)
    stack = TracingLLM(stack)
    return stack
