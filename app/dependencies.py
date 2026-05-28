# app/dependencies.py

from fastapi import Depends

from app.config import settings
from app.llm.factory import get_llm
from app.services.compare_llm_service import CompareLLMService
from app.services.compare_service import CompareService
from app.services.ingest_service import IngestService
from app.services.llm_service import LLMService
from app.services.embedder import get_embeddings
from app.services.query_enhancer import QueryEnhancer
from app.services.query_service import QueryService
from app.services.retrieval import HybridRetriever
from app.services.semantic_cache import SemanticAskCache

_semantic_cache: SemanticAskCache | None = None


def _get_semantic_cache() -> SemanticAskCache | None:
    global _semantic_cache
    if not settings.SEMANTIC_CACHE_ENABLED:
        return None
    if _semantic_cache is None:
        _semantic_cache = SemanticAskCache(
            get_embeddings(),
            enabled=settings.SEMANTIC_CACHE_ENABLED,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
            ttl_seconds=settings.SEMANTIC_CACHE_TTL_SECONDS,
        )
    return _semantic_cache


def get_llm_service() -> LLMService:
    return LLMService(get_llm())


def get_query_enhancer(
    llm_service: LLMService = Depends(get_llm_service),
) -> QueryEnhancer:
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
    return QueryService(
        llm_service=llm_service,
        query_enhancer=query_enhancer,
        semantic_cache=_get_semantic_cache(),
        retriever=HybridRetriever(),
        retrieval_top_k=settings.RETRIEVAL_TOP_K,
        rerank_top_k=settings.RERANK_TOP_K,
    )


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
