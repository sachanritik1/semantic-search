from __future__ import annotations

from typing import Iterable, List, Optional

from langchain_core.documents import Document
from sqlalchemy import JSON, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(512), default="")
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


DATABASE_URL = settings.DATABASE_URL
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


def save_documents(documents: Iterable[Document]) -> int:
    chunks: List[DocumentChunk] = []

    for i, doc in enumerate(documents):
        metadata = dict(doc.metadata) if doc.metadata else {}
        source = str(metadata.get("source", ""))
        chunk_index = int(metadata.get("chunk_index", i))
        chunks.append(
            DocumentChunk(
                source=source,
                chunk_index=chunk_index,
                content=doc.page_content,
                meta=metadata,
            )
        )

    if not chunks:
        return 0

    with SessionLocal() as session:
        session.add_all(chunks)
        session.commit()

    return len(chunks)


def list_chunks(limit: Optional[int] = None) -> List[DocumentChunk]:
    with SessionLocal() as session:
        stmt = select(DocumentChunk).order_by(DocumentChunk.id)
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(session.execute(stmt).scalars().all())
