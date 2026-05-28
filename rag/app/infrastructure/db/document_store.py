from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


def _normalize_database_url(url: str) -> str:
    """Use psycopg v3 for bare postgresql:// URLs (hosting providers omit the driver)."""
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    return url


class Base(DeclarativeBase):
    pass


DATABASE_URL = _normalize_database_url(settings.DATABASE_URL)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    DATABASE_URL,
    future=True,
    connect_args=connect_args,
    pool_pre_ping=not DATABASE_URL.startswith("sqlite"),
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def check_db_connection() -> None:
    """Verify the database is reachable (call at startup)."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
