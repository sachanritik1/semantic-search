from fastapi import APIRouter, Depends

from app.dependencies import get_compare_service
from app.schemas.compare import CompareRequest
from app.services.compare_service import CompareService

router = APIRouter(tags=["query"])


@router.post("/compare")
async def compare_retrievers(
    request: CompareRequest,
    compare_service: CompareService = Depends(get_compare_service),
):
    """Compare dense (RAG) and sparse (BM25) retrievers."""
    return compare_service.compare(request.question, top_k=request.top_k)
