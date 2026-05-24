from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.db.vector_store import (
    DOCUMENT_ID_PAYLOAD_FIELD,
    ensure_payload_indexes,
    upsert_documents,
)


def test_ensure_payload_indexes_skips_when_collection_missing():
    client = MagicMock()
    client.collection_exists.return_value = False

    ensure_payload_indexes(client)

    client.collection_exists.assert_called_once()
    client.create_payload_index.assert_not_called()


def test_ensure_payload_indexes_creates_document_id_index():
    client = MagicMock()
    client.collection_exists.return_value = True

    ensure_payload_indexes(client)

    client.create_payload_index.assert_called_once_with(
        collection_name="semantic-search",
        field_name=DOCUMENT_ID_PAYLOAD_FIELD,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def test_ensure_payload_indexes_ignores_existing_index():
    client = MagicMock()
    client.collection_exists.return_value = True
    client.create_payload_index.side_effect = UnexpectedResponse(
        status_code=409,
        reason_phrase="Conflict",
        content=b"already exists",
        headers={},
    )

    ensure_payload_indexes(client)


def test_ensure_payload_indexes_reraises_unexpected_errors():
    client = MagicMock()
    client.collection_exists.return_value = True
    client.create_payload_index.side_effect = UnexpectedResponse(
        status_code=400,
        reason_phrase="Bad Request",
        content=b"bad request",
        headers={},
    )

    with pytest.raises(UnexpectedResponse):
        ensure_payload_indexes(client)


def test_upsert_documents_ensures_indexes_when_collection_exists():
    embeddings = MagicMock()
    documents = [
        Document(page_content="chunk", metadata={"chunk_id": "chunk-1"}),
    ]
    client = MagicMock()
    client.collection_exists.return_value = True
    vector_store = MagicMock()

    with (
        patch("app.db.vector_store.get_qdrant_client", return_value=client),
        patch("app.db.vector_store.ensure_payload_indexes") as ensure_indexes,
        patch("app.db.vector_store.get_vector_store", return_value=vector_store),
    ):
        upsert_documents(embeddings, documents)

    ensure_indexes.assert_called_once_with(client)
    vector_store.add_documents.assert_called_once_with(documents, ids=["chunk-1"])


def test_upsert_documents_ensures_indexes_after_collection_creation():
    embeddings = MagicMock()
    documents = [
        Document(page_content="chunk", metadata={"chunk_id": "chunk-1"}),
    ]
    client = MagicMock()
    client.collection_exists.return_value = False

    with (
        patch("app.db.vector_store.get_qdrant_client", return_value=client),
        patch("app.db.vector_store.ensure_payload_indexes") as ensure_indexes,
        patch("app.db.vector_store.QdrantVectorStore.from_documents"),
    ):
        upsert_documents(embeddings, documents)

    ensure_indexes.assert_called_once_with(client)
