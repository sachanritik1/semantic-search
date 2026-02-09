# app/services/llm_service.py

from app.llm.base import BaseLLM, LLMResponse


class LLMService:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
      
    ) -> LLMResponse:
        """
        Application-level LLM call.
        """
        return self.llm.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_text_async(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
       
    ) -> LLMResponse:
        return await self.llm.generate_async(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            
        )