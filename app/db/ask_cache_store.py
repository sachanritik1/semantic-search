from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import DateTime, Integer, String, Text, delete, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.db.document_store import Base, SessionLocal

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


def utc_now() -> datetime:
    return datetime.now(_UTC)


def prune_expired(ttl_seconds: int, *, now: datetime | None = None) -> int:
    if ttl_seconds <= 0:
        return 0

    current = now or utc_now()
    cutoff = current - timedelta(seconds=ttl_seconds)
    with SessionLocal() as session:
        result = session.execute(
            delete(AskCacheRow).where(AskCacheRow.created_at < cutoff)
        )
        session.commit()
        return result.rowcount or 0


def list_rows(
    *,
    document_id: str | None = None,
    ttl_seconds: int,
    now: datetime | None = None,
) -> list[AskCacheRow]:
    prune_expired(ttl_seconds, now=now)
    current = now or utc_now()
    cutoff = current - timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None

    with SessionLocal() as session:
        stmt = select(AskCacheRow)
        if document_id is not None:
            stmt = stmt.where(AskCacheRow.document_id == document_id)
        if cutoff is not None:
            stmt = stmt.where(AskCacheRow.created_at >= cutoff)
        stmt = stmt.order_by(AskCacheRow.created_at.desc())
        return list(session.scalars(stmt).all())


def insert_row(
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
        created_at=created_at or utc_now(),
    )
    with SessionLocal() as session:
        session.add(row)
        session.commit()
        session.refresh(row)
        return row
