from fastapi import APIRouter, Depends

from app.dependencies import get_compare_pipeline
from app.api.schemas.compare import CompareRequest
from app.pipelines.compare import ComparePipeline

router = APIRouter(tags=["query"])


@router.post("/compare")
async def compare_retrievers(
    request: CompareRequest,
    pipeline: ComparePipeline = Depends(get_compare_pipeline),
):
    """Compare dense (RAG) and sparse (BM25) retrievers."""
    return await pipeline.compare(request.question, top_k=request.top_k)


@router.post("/compare/llm")
async def compare_retrievers_with_llm(
    request: CompareRequest,
    pipeline: ComparePipeline = Depends(get_compare_pipeline),
):
    """Compare dense and sparse retrievers, then score relevance with an LLM."""
    return await pipeline.compare(
        request.question,
        top_k=request.top_k,
        with_llm=True,
    )
