# app/main.py

import asyncio
import logging
from contextlib import asynccontextmanager

from langfuse import Langfuse, get_client

from app.config import settings
from app.utils.huggingface import configure_hf_hub

configure_hf_hub()

logger = logging.getLogger(__name__)

_log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True,
)
logging.getLogger("app").setLevel(_log_level)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.document_store import check_db_connection
from app.db.vector_store import ensure_payload_indexes
from app.routers import (
    compare,
    enhance,
    health,
    ingest,
    llm,
    prompts,
    query,
    self_consistency,
    tokens,
)


async def _warm_reranker() -> None:
    # Loading the cross-encoder downloads ~80MB from HuggingFace on the first
    # call. Doing it here keeps the latency off the request path so the first
    # /ask doesn't appear to "hang".
    try:
        from app.services.re_ranker import _get_cross_encoder

        await asyncio.to_thread(_get_cross_encoder)
    except Exception:
        logger.exception("Failed to pre-warm cross-encoder; first /ask will be slow")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
        Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    await asyncio.to_thread(check_db_connection)
    await asyncio.to_thread(ensure_payload_indexes)
    await _warm_reranker()
    try:
        yield
    finally:
        get_client().flush()


app = FastAPI(title="RAG API", lifespan=lifespan)

_cors_origins = [
    origin.strip()
    for origin in settings.CORS_ORIGINS.split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(enhance.router)
app.include_router(llm.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(compare.router)
app.include_router(tokens.router)
app.include_router(prompts.router)
app.include_router(self_consistency.router)
