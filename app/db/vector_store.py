from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings

DOCUMENT_ID_PAYLOAD_FIELD = "metadata.document_id"


def _qdrant_client_kwargs() -> dict:
    kwargs: dict = {
        "url": settings.QDRANT_URL,
        "check_compatibility": False,
    }
    if settings.QDRANT_API_KEY:
        kwargs["api_key"] = settings.QDRANT_API_KEY
    return kwargs


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(**_qdrant_client_kwargs())


def get_vector_store(embeddings: Embeddings) -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        **_qdrant_client_kwargs(),
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
                key=DOCUMENT_ID_PAYLOAD_FIELD,
                match=models.MatchValue(value=document_id),
            )
        ]
    )


def ensure_payload_indexes(client: QdrantClient | None = None) -> None:
    """Create payload indexes required for filtered dense retrieval."""
    client = client or get_qdrant_client()
    if not client.collection_exists(settings.QDRANT_COLLECTION_NAME):
        return
    try:
        client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name=DOCUMENT_ID_PAYLOAD_FIELD,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except UnexpectedResponse as exc:
        if exc.status_code != 409:
            raise


def upsert_documents(embeddings: Embeddings, documents: list[Document]) -> None:
    """Upsert documents, creating the collection if missing."""
    if not documents:
        return

    ids = _chunk_ids(documents)
    client = get_qdrant_client()

    if not client.collection_exists(settings.QDRANT_COLLECTION_NAME):
        QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            ids=ids,
            **_qdrant_client_kwargs(),
        )
        ensure_payload_indexes(client)
        return

    ensure_payload_indexes(client)
    vector_store = get_vector_store(embeddings)
    vector_store.add_documents(documents, ids=ids)
