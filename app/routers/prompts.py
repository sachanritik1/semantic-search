from fastapi import APIRouter, Depends

from app.dependencies import get_llm_service
from app.schemas.prompts import PromptTestRequest
from app.services.llm_service import LLMService
from app.utils.prompt_loader import load_prompt, render_prompt

router = APIRouter(tags=["llm"])


@router.post("/prompt/test")
async def test_prompt(
    request: PromptTestRequest,
    llm_service: LLMService = Depends(get_llm_service),
):
    try:
        template = load_prompt(request.template)
        prompt = render_prompt(template, request.variables)
        response = await llm_service.generate_text_async(prompt)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
