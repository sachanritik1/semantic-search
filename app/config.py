# app/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Core
    LLM_PROVIDER: str = "openrouter"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3-flash-preview"

    # OpenRouter
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

    # Langchain
    LANGSMITH_TRACING: bool = True
    LANGSMITH_API_KEY: str = ""

    # Reasoning
    ENABLE_REASONING: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
