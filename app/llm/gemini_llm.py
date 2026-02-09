# app/llm/gemini_llm.py

from app.llm.base import BaseLLM, LLMResponse
from google import genai


class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        
    ) -> LLMResponse:

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        if not response or response.text is None:
            raise ValueError("No text returned from Gemini API")

        return LLMResponse(
            content=response.text,
            model=self.model,
            raw_response=response,
        )
