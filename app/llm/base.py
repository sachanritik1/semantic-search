    # app/llm/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LLMResponse:
    """
    Normalized response returned by any LLM provider.
    """
    content: str
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
       
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Optional token limit

        Returns:
            LLMResponse: Normalized LLM output
        """
        pass

    async def generate_async(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Default async wrapper for sync LLMs.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )