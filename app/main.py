# app/main.py

import logging
import os
from contextlib import asynccontextmanager

from app.config import settings

_log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True,
)
logging.getLogger("app").setLevel(_log_level)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.document_store import init_db
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGCHAIN_PROJECT)
    init_db()
    yield


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
