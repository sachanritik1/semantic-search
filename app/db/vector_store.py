from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "test-2"

client = QdrantClient(url=QDRANT_URL)


def get_vector_store(embeddings: Embeddings) -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        # force_recreate=True,    # recreate collection if it exists
    )


def upsert_documents(embeddings: Embeddings, documents: list[Document]) -> None:
    """Upsert documents, creating the collection if missing."""
    try:
        vector_store = get_vector_store(embeddings)
        vector_store.add_documents(documents)
    except Exception:
        QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
        )
