    # app/llm/base.py

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
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
        model: Optional[str] = None,
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

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream text deltas from the LLM.
        """
        pass

    async def generate_async(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
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
                model=model,
            ),
        )

    async def stream_generate_async(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Default async wrapper that yields sync stream chunks from a thread.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _run() -> None:
            try:
                for chunk in self.stream_generate(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        await loop.run_in_executor(None, _run)
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk