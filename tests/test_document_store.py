import pytest
from langchain_core.documents import Document
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import document_store
from app.utils.ids import new_document_id, stamp_document_chunks


@pytest.fixture
def isolated_store(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

    monkeypatch.setattr(document_store, "engine", engine)
    monkeypatch.setattr(document_store, "SessionLocal", session_factory)
    document_store.init_db()
    yield document_store


def test_save_documents_requires_identity_metadata(isolated_store):
    with pytest.raises(ValueError, match="document_id and chunk_id"):
        isolated_store.save_documents([Document(page_content="orphan")])


def test_save_documents_persists_identity_fields(isolated_store):
    document_id = new_document_id()
    chunks = [Document(page_content="chunk body", metadata={})]
    stamp_document_chunks(chunks, document_id=document_id, source="doc.pdf")

    saved = isolated_store.save_documents(chunks)
    assert saved == 1

    rows = isolated_store.list_chunks()
    assert len(rows) == 1
    row = rows[0]
    assert row.document_id == document_id
    assert row.chunk_id == chunks[0].metadata["chunk_id"]
    assert row.status == "active"
    assert row.content_hash
    assert row.tenant_id == "default"

    doc = isolated_store.chunk_to_document(row)
    assert doc.metadata["chunk_id"] == row.chunk_id


def test_list_chunks_active_only(isolated_store):
    document_id = new_document_id()
    chunks = [Document(page_content="x", metadata={})]
    stamp_document_chunks(chunks, document_id=document_id, source="a.pdf")
    isolated_store.save_documents(chunks)

    with isolated_store.SessionLocal() as session:
        row = session.execute(select(isolated_store.DocumentChunk)).scalar_one()
        row.status = isolated_store.CHUNK_STATUS_DELETED
        session.commit()

    assert isolated_store.list_chunks(active_only=True) == []
    assert len(isolated_store.list_chunks(active_only=False)) == 1


def test_list_chunks_for_document_filters_by_document_id(isolated_store):
    doc_a = new_document_id()
    doc_b = new_document_id()
    chunks_a = [Document(page_content="a1", metadata={}), Document(page_content="a2", metadata={})]
    chunks_b = [Document(page_content="b1", metadata={})]
    stamp_document_chunks(chunks_a, document_id=doc_a, source="a.pdf")
    stamp_document_chunks(chunks_b, document_id=doc_b, source="b.pdf")
    isolated_store.save_documents(chunks_a)
    isolated_store.save_documents(chunks_b)

    rows = isolated_store.list_chunks_for_document(doc_a)
    assert len(rows) == 2
    assert all(row.document_id == doc_a for row in rows)
    assert [row.chunk_index for row in rows] == [0, 1]
