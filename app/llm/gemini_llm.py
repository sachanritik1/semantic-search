# app/llm/gemini_llm.py

from app.llm.base import BaseLLM, LLMResponse
from google import genai
from google.genai import types



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
        model: str | None = None,
    ) -> LLMResponse:
        use_model = model or self.model

        response = self.client.models.generate_content( # type: ignore
            model=use_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        if not response or response.text is None:
            raise ValueError("No text returned from Gemini API")
        
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
        return LLMResponse(
            content=response.text,
            model=use_model,
            raw_response=response,
            usage=usage,
        )
