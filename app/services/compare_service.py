from app.db.document_store import list_chunks
from app.db.weaviate_store import bm25_search, dense_search
from app.services.embedder import get_embeddings


class CompareService:
    def compare(self, question: str, top_k: int = 5) -> dict:
        embeddings = get_embeddings()
        query_embedding = embeddings.embed_query(question)

        dense_results_raw = dense_search(query_embedding, limit=top_k)
        dense_results = [
            {
                "index": i,
                "content": doc.page_content,
                "metadata": getattr(doc, "metadata", None) or {},
            }
            for i, doc in enumerate([d for d, _ in dense_results_raw])
        ]

        chunks = list(list_chunks())
        if not chunks:
            return {"dense": dense_results, "sparse": []}

        sparse_results_raw = bm25_search(question, limit=top_k)

        sparse_results = [
            {
                "index": i,
                "score": score,
                "content": doc.page_content,
                "document_id": doc.metadata.get("document_id", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "source": doc.metadata.get("source", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "metadata": doc.metadata or {},
            }
            for i, (doc, score) in enumerate(sparse_results_raw)
        ]

        return {"dense": dense_results, "sparse": sparse_results}
