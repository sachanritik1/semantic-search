from langchain_core.documents import Document

from app.db.document_store import chunk_to_document, list_chunks
from app.services.dense_retriever import DenseRetriever
from app.services.embedder import embeddings
from app.services.llm_service import LLMService
from app.services.query_enhancer import QueryEnhancer
from app.services.re_ranker import re_rank_docs
from app.services.sparse_retriever import SparseRetriever
from app.utils.prompts import build_prompt


class QueryService:
    def __init__(
        self,
        llm_service: LLMService,
        query_enhancer: QueryEnhancer,
    ):
        self.llm_service = llm_service
        self.query_enhancer = query_enhancer

    async def ask(self, question: str) -> dict:
        query = self.query_enhancer.enhance(question) or question

        dense_ranked = await self._retrieve_and_rerank_dense(query)
        sparse_ranked = await self._retrieve_and_rerank_sparse(query)
        combined_docs = self._merge_documents(dense_ranked, sparse_ranked)

        prompt_text = build_prompt(docs=combined_docs, question=query)
        response = self.llm_service.generate_text(prompt_text)

        return {
            "response": response.content,
            "original_question": question,
            "enhanced_question": query,
        }

    async def _retrieve_and_rerank_dense(self, query: str) -> list[Document]:
        dense = DenseRetriever(embeddings, default_k=10)
        dense_docs = dense.retrieve(query)
        print(f"Retrieved {len(dense_docs)} dense documents.")
        return await self._rerank_or_fallback(query, dense_docs, label="Dense")

    async def _retrieve_and_rerank_sparse(self, query: str) -> list[Document]:
        chunks = list_chunks()
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        sparse = SparseRetriever()
        sparse.build_index(texts)
        sparse_res = sparse.query(query, top_k=10)
        sparse_docs = [chunk_to_document(chunks[idx]) for idx, _, _ in sparse_res]
        print(f"Retrieved {len(sparse_docs)} sparse documents.")

        if not sparse_docs:
            return []
        return await self._rerank_or_fallback(query, sparse_docs, label="Sparse")

    async def _rerank_or_fallback(
        self,
        query: str,
        docs: list[Document],
        *,
        label: str,
    ) -> list[Document]:
        try:
            return await re_rank_docs(
                query,
                docs,
                llm_service=self.llm_service,
                top_n=5,
                max_candidates=8,
                max_doc_chars=200,
                max_tokens=64,
                timeout_s=12.0,
                batch_count=2,
            )
        except (TimeoutError, ValueError) as exc:
            print(f"{label} rerank failed: {exc}")
            return docs[:5]

    @staticmethod
    def _merge_documents(
        dense_ranked: list[Document],
        sparse_ranked: list[Document],
    ) -> list[Document]:
        combined: list[Document] = []
        seen: set[str] = set()
        for doc in dense_ranked + sparse_ranked:
            meta = doc.metadata or {}
            key = str(meta.get("chunk_id") or doc.page_content.strip())
            if key in seen:
                continue
            seen.add(key)
            combined.append(doc)
        return combined
