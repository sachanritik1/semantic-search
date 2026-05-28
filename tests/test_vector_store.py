from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.adapters.vector_store import upsert_documents


def test_upsert_documents_skips_empty_list():
    with patch("app.db.weaviate_store.get_weaviate_client") as mock_client:
        upsert_documents([], [])
    mock_client.assert_not_called()


def test_upsert_documents_calls_weaviate_batch():
    embeddings = [[0.1, 0.2]]
    documents = [
        Document(page_content="chunk", metadata={"chunk_id": "chunk-1", "document_id": "doc-1"}),
    ]

    mock_batch = MagicMock()
    mock_batch.__enter__ = MagicMock(return_value=mock_batch)
    mock_batch.__exit__ = MagicMock(return_value=None)

    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    mock_client.batch.dynamic.return_value = mock_batch
    mock_client.batch.failed_objects = []

    with (
        patch("app.db.weaviate_store.get_weaviate_client", return_value=mock_client),
        patch("app.db.weaviate_store.ensure_collection", return_value=mock_client),
        patch("app.db.weaviate_store.settings.WEAVIATE_GRPC_ENABLED", True),
        patch("app.db.weaviate_store._grpc_available", True),
    ):
        upsert_documents(embeddings, documents)

    mock_client.batch.dynamic.assert_called_once()
    mock_batch.add_object.assert_called_once()
    call_kwargs = mock_batch.add_object.call_args.kwargs
    assert call_kwargs["properties"]["content"] == "chunk"
    assert call_kwargs["properties"]["document_id"] == "doc-1"
    assert call_kwargs["vector"] == [0.1, 0.2]
    assert call_kwargs["collection"] == "DocumentChunk"


def test_upsert_documents_falls_back_to_rest_when_grpc_disabled():
    embeddings = [[0.1, 0.2]]
    documents = [
        Document(page_content="chunk", metadata={"chunk_id": "chunk-1", "document_id": "doc-1"}),
    ]

    mock_col = MagicMock()
    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    mock_client.collections.use.return_value = mock_col

    with (
        patch("app.db.weaviate_store.get_weaviate_client", return_value=mock_client),
        patch("app.db.weaviate_store.ensure_collection", return_value=mock_client),
        patch("app.db.weaviate_store.settings.WEAVIATE_GRPC_ENABLED", False),
        patch("app.db.weaviate_store._grpc_available", False),
    ):
        upsert_documents(embeddings, documents)

    mock_client.batch.dynamic.assert_not_called()
    mock_col.data.insert.assert_called_once()
    assert mock_col.data.insert.call_args.kwargs["vector"] == [0.1, 0.2]
