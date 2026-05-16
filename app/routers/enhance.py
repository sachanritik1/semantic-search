from fastapi import APIRouter, Depends

from app.dependencies import get_query_enhancer
from app.schemas.common import EnhanceResponse, QuestionRequest
from app.services.query_enhancer import QueryEnhancer

router = APIRouter(tags=["query"])


@router.post("/enhance", response_model=EnhanceResponse)
def enhance_query(
    request: QuestionRequest,
    query_enhancer: QueryEnhancer = Depends(get_query_enhancer),
):
    """Rewrite a user query for better retrieval (standalone test endpoint)."""
    enhanced = query_enhancer.enhance(request.question) or request.question
    return EnhanceResponse(original=request.question, enhanced=enhanced)
