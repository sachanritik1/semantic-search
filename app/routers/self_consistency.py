from fastapi import APIRouter, Depends

from app.dependencies import get_llm_service
from app.schemas.common import QuestionRequest
from app.services.llm_service import LLMService
from app.services.self_consistency import generate_with_self_consistency

router = APIRouter(tags=["llm"])


@router.post("/self-consistency")
async def self_consistency_test(
    request: QuestionRequest,
    llm_service: LLMService = Depends(get_llm_service),
):
    final_answer = await generate_with_self_consistency(
        llm_service=llm_service,
        prompt=request.question,
        runs=5,
    )
    return {"final_answer": final_answer}
