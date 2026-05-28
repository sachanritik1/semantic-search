from typing import Protocol

from langchain_core.documents import Document

from app.adapters.vector_store import document_has_chunks, hybrid_search
from app.adapters.embedder import get_embeddings


class Retriever(Protocol):
    def retrieve(
        self,
        queries: list[str],
        document_id: str,
        limit: int = 10,
    ) -> list[tuple[Document, float]]: ...


class HybridRetriever:
    """Dense + sparse hybrid retrieval backed by Weaviate."""

    def __init__(self, embedder=None) -> None:
        self._embeddings = embedder or get_embeddings()

    def retrieve(
        self,
        queries: list[str],
        document_id: str,
        limit: int = 10,
    ) -> list[tuple[Document, float]]:
        if not document_has_chunks(document_id):
            return []

        all_hits: dict[str, tuple[Document, float]] = {}

        for q in queries:
            query_embedding = self._embeddings.embed_query(q)
            results = hybrid_search(
                q,
                query_embedding,
                document_id=document_id,
                limit=limit,
            )
            for doc, score in results:
                key = doc.metadata.get("chunk_id", doc.page_content)
                if key not in all_hits or score > all_hits[key][1]:
                    all_hits[key] = (doc, score)

        fused = sorted(all_hits.values(), key=lambda x: x[1], reverse=True)
        return fused[:limit]
