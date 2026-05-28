from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import cast

from sqlalchemy import DateTime, Integer, String, Text, delete, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.infrastructure.db.document_store import Base, SessionLocal

_UTC = timezone.utc
_JSON = JSON().with_variant(JSONB, "postgresql")


class AskCacheRow(Base):
    __tablename__ = "ask_cache_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(36), index=True)
    original_question: Mapped[str] = mapped_column(Text)
    enhanced_question: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(_JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        default=lambda: datetime.now(_UTC),
    )


class AskCacheStore:
    """Adapter for PostgreSQL semantic cache storage."""

    def __init__(self, session_factory=SessionLocal) -> None:
        self._session_factory = session_factory

    def _utc_now(self) -> datetime:
        return datetime.now(_UTC)

    def prune_expired(self, ttl_seconds: int, *, now: datetime | None = None) -> int:
        if ttl_seconds <= 0:
            return 0

        current = now or self._utc_now()
        cutoff = current - timedelta(seconds=ttl_seconds)
        with self._session_factory() as session:
            result = session.execute(
                delete(AskCacheRow).where(AskCacheRow.created_at < cutoff)
            )
            session.commit()
            rowcount = cast(CursorResult[object], result).rowcount
            return rowcount or 0

    def list_rows(
        self,
        *,
        document_id: str | None = None,
        ttl_seconds: int,
        now: datetime | None = None,
    ) -> list[AskCacheRow]:
        self.prune_expired(ttl_seconds, now=now)
        current = now or self._utc_now()
        cutoff = current - timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None

        with self._session_factory() as session:
            stmt = select(AskCacheRow)
            if document_id is not None:
                stmt = stmt.where(AskCacheRow.document_id == document_id)
            if cutoff is not None:
                stmt = stmt.where(AskCacheRow.created_at >= cutoff)
            stmt = stmt.order_by(AskCacheRow.created_at.desc())
            return list(session.scalars(stmt).all())

    def insert(
        self,
        *,
        document_id: str,
        original_question: str,
        enhanced_question: str,
        response: str,
        embedding: list[float],
        created_at: datetime | None = None,
    ) -> AskCacheRow:
        row = AskCacheRow(
            document_id=document_id,
            original_question=original_question,
            enhanced_question=enhanced_question,
            response=response,
            embedding=embedding,
            created_at=created_at or self._utc_now(),
        )
        with self._session_factory() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return row


# Backward-compatible module-level functions delegate to a default instance
_default_store: AskCacheStore | None = None


def _default_cache_store() -> AskCacheStore:
    global _default_store
    if _default_store is None:
        _default_store = AskCacheStore()
    return _default_store


def utc_now() -> datetime:
    return _default_cache_store()._utc_now()


def prune_expired(ttl_seconds: int, *, now: datetime | None = None) -> int:
    return _default_cache_store().prune_expired(ttl_seconds, now=now)


def list_rows(
    *,
    document_id: str | None = None,
    ttl_seconds: int,
    now: datetime | None = None,
) -> list[AskCacheRow]:
    return _default_cache_store().list_rows(
        document_id=document_id,
        ttl_seconds=ttl_seconds,
        now=now,
    )


def insert_row(
    *,
    document_id: str,
    original_question: str,
    enhanced_question: str,
    response: str,
    embedding: list[float],
    created_at: datetime | None = None,
) -> AskCacheRow:
    return _default_cache_store().insert(
        document_id=document_id,
        original_question=original_question,
        enhanced_question=enhanced_question,
        response=response,
        embedding=embedding,
        created_at=created_at,
    )


__all__ = [
    "AskCacheRow",
    "AskCacheStore",
    "insert_row",
    "list_rows",
    "prune_expired",
    "utc_now",
]
