# app/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Core
    LOG_LEVEL: str = "INFO"
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

    # Langfuse (tracing toggled via LANGFUSE_TRACING_ENABLED env var — Langfuse SDK)
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://us.cloud.langfuse.com"

    # Query enhancement (optional override; defaults to provider model)
    ENHANCER_MODEL: str | None = "openai/gpt-oss-20b:free"

    # Reasoning
    ENABLE_REASONING: bool = False

    # Database (PostgreSQL; run `alembic upgrade head` after `docker compose up -d`)
    DATABASE_URL: str = (
        "postgresql+psycopg://semantic:semantic@localhost:5432/semantic_search"
    )

    # Hugging Face Hub (optional; avoids unauthenticated download warnings)
    HF_TOKEN: str = ""

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "semantic-search"
    QDRANT_API_KEY: str = ""

    # Multi-tenant (single-tenant default)
    DEFAULT_TENANT_ID: str = "default"

    # CORS (comma-separated origins for browser clients)
    CORS_ORIGINS: str = "http://localhost:3000"

    # Hybrid retrieval / ask pipeline
    DENSE_WEIGHT: float = 0.5
    SPARSE_WEIGHT: float = 0.5
    RETRIEVAL_TOP_K: int = 10
    FUSION_MIN_SCORE: float = 0.25
    FUSION_MIN_DOCS: int = 10
    RERANK_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = 8
    RERANK_MIN_RELEVANCE: float = 4.0

    # SSE streaming (/ask/stream)
    SSE_HEARTBEAT_INTERVAL_S: float = 15.0

    # Provider-native prompt prefix caching (OpenAI / Gemini / OpenRouter)
    PROMPT_CACHE_ENABLED: bool = True
    PROMPT_CACHE_VERSION: str = "v1"

    # Semantic /ask response cache (in-process, TTL-only eviction)
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_THRESHOLD: float = 0.85
    SEMANTIC_CACHE_TTL_SECONDS: int = 3600

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
