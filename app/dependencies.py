# app/dependencies.py

from app.llm.factory import get_llm
from app.services.llm_service import LLMService


def get_llm_service() -> LLMService:
    return LLMService(get_llm())
