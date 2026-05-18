from fastapi import APIRouter, Depends

from app.dependencies import get_query_service
from app.schemas.common import QuestionRequest
from app.services.query_service import QueryService

router = APIRouter(tags=["query"])


@router.post("/ask")
async def ask_question(
    request: QuestionRequest,
    query_service: QueryService = Depends(get_query_service),
):
    return await query_service.ask(
        request.question,
        document_id=request.document_id,
    )
