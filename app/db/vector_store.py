from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings

client = QdrantClient(url=settings.QDRANT_URL)


def get_vector_store(embeddings: Embeddings) -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
        create_collection_if_missing=True,
    )


def _chunk_ids(documents: list[Document]) -> list[str]:
    ids: list[str] = []
    for doc in documents:
        chunk_id = (doc.metadata or {}).get("chunk_id")
        if not chunk_id:
            raise ValueError("Each document must have chunk_id in metadata before upsert")
        ids.append(str(chunk_id))
    return ids


def document_id_filter(document_id: str) -> models.Filter:
    """Filter points by document_id in LangChain Qdrant payload metadata."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
    )


def upsert_documents(embeddings: Embeddings, documents: list[Document]) -> None:
    """Upsert documents, creating the collection if missing."""
    if not documents:
        return

    ids = _chunk_ids(documents)
    try:
        vector_store = get_vector_store(embeddings)
        vector_store.add_documents(documents, ids=ids)
    except Exception:
        QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            url=settings.QDRANT_URL,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            ids=ids,
        )
