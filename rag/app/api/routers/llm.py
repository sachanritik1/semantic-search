from fastapi import APIRouter, Depends

from app.dependencies import get_llm_service
from app.api.schemas.common import QuestionRequest
from app.domain.llm_service import LLMService

router = APIRouter(tags=["llm"])


@router.post("/llm/test")
def test_llm(
    request: QuestionRequest,
    llm_service: LLMService = Depends(get_llm_service),
):
    response = llm_service.generate_text(request.question)
    return {"response": response}
