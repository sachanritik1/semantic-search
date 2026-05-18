import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import ask_cache_store, document_store


@pytest.fixture
def isolated_ask_cache_db(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        future=True,
    )

    monkeypatch.setattr(document_store, "engine", engine)
    monkeypatch.setattr(document_store, "SessionLocal", session_factory)
    monkeypatch.setattr(ask_cache_store, "engine", engine)
    monkeypatch.setattr(ask_cache_store, "SessionLocal", session_factory)
    document_store.init_db()
