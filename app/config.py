# app/config.py

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Core
    LLM_PROVIDER: str = Field(default="gemini")

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"

    # Gemini
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Langchain
    LANGSMITH_TRACING:str
    LANGSMITH_API_KEY:str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
