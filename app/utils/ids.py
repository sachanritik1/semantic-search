from __future__ import annotations

import hashlib
import uuid

from langchain_core.documents import Document

from app.config import settings

_CHUNK_NAMESPACE = uuid.NAMESPACE_DNS


def new_document_id() -> str:
    return str(uuid.uuid4())


def chunk_id_for(document_id: str, chunk_index: int) -> str:
    """Deterministic UUID shared by SQLite and Qdrant."""
    name = f"{document_id}:{chunk_index}"
    return str(uuid.uuid5(_CHUNK_NAMESPACE, name))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stamp_document_chunks(
    chunks: list[Document],
    *,
    document_id: str,
    source: str,
    tenant_id: str | None = None,
) -> list[Document]:
    tenant = tenant_id or settings.DEFAULT_TENANT_ID
    for i, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        metadata.update(
            {
                "document_id": document_id,
                "chunk_id": chunk_id_for(document_id, i),
                "chunk_index": i,
                "source": source,
                "tenant_id": tenant,
            }
        )
        chunk.metadata = metadata
    return chunks
