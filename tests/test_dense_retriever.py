from unittest.mock import MagicMock, patch

from qdrant_client.http import models

from app.db.vector_store import document_id_filter
from app.services.dense_retriever import DenseRetriever


def test_document_id_filter_uses_qdrant_models():
    qdrant_filter = document_id_filter("doc-1")
    assert isinstance(qdrant_filter, models.Filter)
    assert len(qdrant_filter.must) == 1
    condition = qdrant_filter.must[0]
    assert condition.key == "metadata.document_id"
    assert condition.match.value == "doc-1"


def test_retrieve_with_scores_passes_document_filter():
    embeddings = MagicMock()
    vector_store = MagicMock()
    vector_store.similarity_search_with_relevance_scores.return_value = []
    qdrant_filter = document_id_filter("doc-1")

    retriever = DenseRetriever(embeddings, default_k=5)

    with patch(
        "app.services.dense_retriever.get_vector_store",
        return_value=vector_store,
    ):
        retriever.retrieve_with_scores("query", k=3, document_id="doc-1")

    vector_store.similarity_search_with_relevance_scores.assert_called_once_with(
        "query",
        k=3,
        filter=qdrant_filter,
    )


def test_retrieve_with_scores_omits_filter_when_unscoped():
    embeddings = MagicMock()
    vector_store = MagicMock()
    vector_store.similarity_search_with_relevance_scores.return_value = []

    retriever = DenseRetriever(embeddings, default_k=5)

    with patch(
        "app.services.dense_retriever.get_vector_store",
        return_value=vector_store,
    ):
        retriever.retrieve_with_scores("query", k=3)

    vector_store.similarity_search_with_relevance_scores.assert_called_once_with(
        "query",
        k=3,
        filter=None,
    )
