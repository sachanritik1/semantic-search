

from openai import OpenAI
from app.llm.base import BaseLLM, LLMResponse


class OpenaiLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
       
    ) -> LLMResponse:

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            
        )
        
        if not response or not response.output_text:
            raise ValueError("No text returned from OpenAI API")
        
        return LLMResponse(
            content=response.output_text,
            model=self.model,
            raw_response=response,
        )