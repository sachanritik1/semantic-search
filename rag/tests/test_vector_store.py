from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.adapters.vector_store import VectorStore, upsert_documents


def test_upsert_documents_skips_empty_list():
    with patch("app.adapters.vector_store.get_weaviate_client") as mock_client:
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

    store = VectorStore(
        url="http://localhost:8080",
        grpc_enabled=True,
        collection_name="DocumentChunk",
    )
    store._client = mock_client
    store.upsert(embeddings, documents)

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

    store = VectorStore(
        url="http://localhost:8080",
        grpc_enabled=False,
        collection_name="DocumentChunk",
    )
    store._client = mock_client
    store.upsert(embeddings, documents)

    mock_client.batch.dynamic.assert_not_called()
    mock_col.data.insert.assert_called_once()
    assert mock_col.data.insert.call_args.kwargs["vector"] == [0.1, 0.2]
