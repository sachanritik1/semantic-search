from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.embeddings import Embeddings

client = QdrantClient(url="http://localhost:6333")

def get_vector_store(embeddings: Embeddings) -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="test-2",
    # force_recreate=True,    # recreate collection if it exists
)

