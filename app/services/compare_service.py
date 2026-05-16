from app.db.document_store import list_chunks
from app.services.dense_retriever import DenseRetriever
from app.services.embedder import embeddings
from app.services.sparse_retriever import SparseRetriever


class CompareService:
    def compare(self, question: str, top_k: int = 5) -> dict:
        dense = DenseRetriever(embeddings, default_k=top_k)
        dense_docs = dense.retrieve(question, k=top_k)

        dense_results = [
            {
                "index": i,
                "content": doc.page_content,
                "metadata": getattr(doc, "metadata", None) or {},
            }
            for i, doc in enumerate(dense_docs)
        ]

        chunks = list_chunks()
        texts = [c.content for c in chunks]
        if not texts:
            return {"dense": dense_results, "sparse": []}

        sparse = SparseRetriever()
        sparse.build_index(texts)
        sparse_res = sparse.query(question, top_k=top_k)
        sparse_results = [
            {
                "index": idx,
                "score": score,
                "content": text,
                "document_id": chunks[idx].document_id,
                "chunk_id": chunks[idx].chunk_id,
                "source": chunks[idx].source,
                "chunk_index": chunks[idx].chunk_index,
                "metadata": chunks[idx].meta,
            }
            for idx, score, text in sparse_res
        ]

        return {"dense": dense_results, "sparse": sparse_results}
