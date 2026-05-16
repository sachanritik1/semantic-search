# app/dependencies.py

from fastapi import Depends

from app.llm.factory import get_llm
from app.services.compare_llm_service import CompareLLMService
from app.services.compare_service import CompareService
from app.services.ingest_service import IngestService
from app.services.llm_service import LLMService
from app.services.query_enhancer import QueryEnhancer
from app.services.query_service import QueryService


def get_llm_service() -> LLMService:
    return LLMService(get_llm())


def get_query_enhancer(llm_service: LLMService = Depends(get_llm_service)) -> QueryEnhancer:
    from app.config import settings

    return QueryEnhancer(
        llm_service=llm_service,
        enhancer_model=settings.ENHANCER_MODEL,
    )


def get_ingest_service() -> IngestService:
    return IngestService()


def get_query_service(
    llm_service: LLMService = Depends(get_llm_service),
    query_enhancer: QueryEnhancer = Depends(get_query_enhancer),
) -> QueryService:
    return QueryService(llm_service=llm_service, query_enhancer=query_enhancer)


def get_compare_service() -> CompareService:
    return CompareService()


def get_compare_llm_service(
    compare_service: CompareService = Depends(get_compare_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> CompareLLMService:
    return CompareLLMService(
        compare_service=compare_service,
        llm_service=llm_service,
    )
