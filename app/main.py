# app/main.py

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
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
        os.environ.setdefault("LANGCHAIN_PROJECT", "semantic-search")
    init_db()
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


app.include_router(health.router)
app.include_router(enhance.router)
app.include_router(llm.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(compare.router)
app.include_router(tokens.router)
app.include_router(prompts.router)
app.include_router(self_consistency.router)
