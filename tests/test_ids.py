from langchain_core.documents import Document

from app.utils.ids import (
    chunk_id_for,
    content_hash,
    new_document_id,
    stamp_document_chunks,
)


def test_chunk_id_is_deterministic():
    doc_id = "550e8400-e29b-41d4-a716-446655440000"
    assert chunk_id_for(doc_id, 0) == chunk_id_for(doc_id, 0)
    assert chunk_id_for(doc_id, 0) != chunk_id_for(doc_id, 1)


def test_content_hash_stable():
    text = "hello world"
    assert content_hash(text) == content_hash(text)
    assert content_hash(text) != content_hash("other")


def test_new_document_id_is_deterministic_from_file_bytes():
    same = b"same pdf content"
    different = b"different pdf content"
    assert new_document_id(same) == new_document_id(same)
    assert new_document_id(same) != new_document_id(different)


def test_stamp_document_chunks_sets_identity_fields():
    document_id = new_document_id(b"some pdf bytes")
    chunks = [
        Document(page_content="first", metadata={}),
        Document(page_content="second", metadata={}),
    ]
    stamp_document_chunks(chunks, document_id=document_id, source="test.pdf")

    assert chunks[0].metadata["document_id"] == document_id
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["source"] == "test.pdf"
    assert chunks[0].metadata["tenant_id"] == "default"
    assert chunks[0].metadata["chunk_id"] == chunk_id_for(document_id, 0)
    assert chunks[1].metadata["chunk_id"] == chunk_id_for(document_id, 1)
