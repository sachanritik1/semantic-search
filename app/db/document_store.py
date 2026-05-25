from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Optional

from langchain_core.documents import Document
from sqlalchemy import DateTime, Integer, String, Text, create_engine, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.types import JSON

from app.config import settings
from app.utils.ids import content_hash

CHUNK_STATUS_ACTIVE = "active"
CHUNK_STATUS_DELETED = "deleted"

_JSON = JSON().with_variant(JSONB, "postgresql")


def _normalize_database_url(url: str) -> str:
    """Use psycopg v3 for bare postgresql:// URLs (hosting providers omit the driver)."""
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    return url


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(36), index=True)
    chunk_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    source: Mapped[str] = mapped_column(String(512), default="")
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    status: Mapped[str] = mapped_column(
        String(16), default=CHUNK_STATUS_ACTIVE, index=True
    )
    meta: Mapped[dict] = mapped_column("metadata", _JSON, default=dict)
    embedding: Mapped[list[float]] = mapped_column(_JSON, default=list)


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


def chunk_to_document(chunk: DocumentChunk) -> Document:
    metadata = dict(chunk.meta or {})
    metadata.setdefault("source", chunk.source)
    metadata.setdefault("chunk_index", chunk.chunk_index)
    metadata.setdefault("document_id", chunk.document_id)
    metadata.setdefault("chunk_id", chunk.chunk_id)
    metadata.setdefault("tenant_id", chunk.tenant_id)
    return Document(page_content=chunk.content, metadata=metadata)


def save_documents(documents: Iterable[Document]) -> int:
    chunks: List[DocumentChunk] = []
    now = datetime.now(timezone.utc)

    for i, doc in enumerate(documents):
        metadata = dict(doc.metadata) if doc.metadata else {}
        document_id = metadata.get("document_id")
        chunk_id = metadata.get("chunk_id")
        if not document_id or not chunk_id:
            raise ValueError(
                "Each chunk must have document_id and chunk_id in metadata before save"
            )

        source = str(metadata.get("source", ""))
        chunk_index = int(metadata.get("chunk_index", i))
        tenant_id = str(metadata.get("tenant_id", settings.DEFAULT_TENANT_ID))

        chunks.append(
            DocumentChunk(
                document_id=str(document_id),
                chunk_id=str(chunk_id),
                tenant_id=tenant_id,
                source=source,
                chunk_index=chunk_index,
                content=doc.page_content,
                content_hash=content_hash(doc.page_content),
                ingested_at=now,
                status=CHUNK_STATUS_ACTIVE,
                meta=metadata,
                embedding=[],
            )
        )

    if not chunks:
        return 0

    with SessionLocal() as session:
        session.add_all(chunks)
        session.commit()

    return len(chunks)


def list_chunks(
    limit: Optional[int] = None, *, active_only: bool = True
) -> List[DocumentChunk]:
    with SessionLocal() as session:
        stmt = select(DocumentChunk).order_by(DocumentChunk.id)
        if active_only:
            stmt = stmt.where(DocumentChunk.status == CHUNK_STATUS_ACTIVE)
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(session.execute(stmt).scalars().all())


def list_chunks_for_document(
    document_id: str,
    *,
    active_only: bool = True,
) -> List[DocumentChunk]:
    with SessionLocal() as session:
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        if active_only:
            stmt = stmt.where(DocumentChunk.status == CHUNK_STATUS_ACTIVE)
        return list(session.execute(stmt).scalars().all())
