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
    OPENROUTER_MODEL: str = "openai/gpt-oss-120b:free"

    # Langchain
    LANGSMITH_TRACING: bool = True
    LANGSMITH_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "semantic-search"

    # Reasoning
    ENABLE_REASONING: bool = False

    # Database
    DATABASE_URL: str = "sqlite:///./docstore.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
