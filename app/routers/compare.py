from fastapi import APIRouter, Depends

from app.dependencies import get_compare_llm_service, get_compare_service
from app.schemas.compare import CompareRequest
from app.services.compare_llm_service import CompareLLMService
from app.services.compare_service import CompareService

router = APIRouter(tags=["query"])


@router.post("/compare")
async def compare_retrievers(
    request: CompareRequest,
    compare_service: CompareService = Depends(get_compare_service),
):
    """Compare dense (RAG) and sparse (BM25) retrievers."""
    return compare_service.compare(request.question, top_k=request.top_k)


@router.post("/compare/llm")
async def compare_retrievers_with_llm(
    request: CompareRequest,
    compare_llm_service: CompareLLMService = Depends(get_compare_llm_service),
):
    """Compare dense and sparse retrievers, then score relevance with an LLM."""
    return await compare_llm_service.compare_with_llm(
        request.question,
        top_k=request.top_k,
    )
