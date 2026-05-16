# app/dependencies.py

from fastapi import Depends

from app.llm.factory import get_llm
from app.services.llm_service import LLMService
from app.services.query_enhancer import QueryEnhancer


def get_llm_service() -> LLMService:
    return LLMService(get_llm())


def get_query_enhancer(llm_service: LLMService = Depends(get_llm_service)) -> QueryEnhancer:
    from app.config import settings

    return QueryEnhancer(
        llm_service=llm_service,
        enhancer_model=settings.ENHANCER_MODEL,
    )
