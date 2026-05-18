import logging

from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)
from app.db.document_store import (
    chunk_to_document,
    list_chunks,
    list_chunks_for_document,
)
from app.services.dense_retriever import DenseRetriever
from app.services.document_fusion import fuse_documents
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

    async def ask(self, question: str, *, document_id: str | None = None) -> dict:
        query = self.query_enhancer.enhance(question) or question

        if document_id and not list_chunks_for_document(document_id):
            logger.info(
                "No chunks found for document_id=%s; answering without retrieval context",
                document_id,
            )
            prompt_text = build_prompt(docs=[], question=query)
            response = self.llm_service.generate_text(prompt_text)
            return {
                "response": response.content,
                "original_question": question,
                "enhanced_question": query,
            }

        dense_hits = self._retrieve_dense(query, document_id=document_id)
        sparse_hits = self._retrieve_sparse(query, document_id=document_id)

        fused = fuse_documents(
            dense_hits,
            sparse_hits,
            dense_weight=settings.DENSE_WEIGHT,
            sparse_weight=settings.SPARSE_WEIGHT,
        )[: settings.FUSION_TOP_K]

        if not fused:
            prompt_text = build_prompt(docs=[], question=query)
            response = self.llm_service.generate_text(prompt_text)
            return {
                "response": response.content,
                "original_question": question,
                "enhanced_question": query,
            }

        rerank_result = re_rank_docs(
            query,
            fused,
            top_n=settings.RERANK_TOP_K,
        )

        if rerank_result.failed:
            logger.warning(
                "Rerank failed; using all %d fused documents for answer context",
                len(fused),
            )
            answer_docs = fused
        else:
            answer_docs = rerank_result.docs

        prompt_text = build_prompt(
            docs=answer_docs,
            question=question,
            search_query=query,
        )
        response = self.llm_service.generate_text(prompt_text)

        return {
            "response": response.content,
            "original_question": question,
            "enhanced_question": query,
        }

    def _retrieve_dense(
        self,
        query: str,
        *,
        document_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        dense = DenseRetriever(embeddings, default_k=settings.RETRIEVAL_TOP_K)
        hits = dense.retrieve_with_scores(
            query,
            k=settings.RETRIEVAL_TOP_K,
            document_id=document_id,
        )
        print(f"Retrieved {len(hits)} dense documents.")
        return hits

    def _retrieve_sparse(
        self,
        query: str,
        *,
        document_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        if document_id:
            chunks = list_chunks_for_document(document_id)
        else:
            chunks = list_chunks()
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        sparse = SparseRetriever()
        sparse.build_index(texts)
        sparse_res = sparse.query(query, top_k=settings.RETRIEVAL_TOP_K)
        hits = [
            (chunk_to_document(chunks[idx]), score) for idx, score, _ in sparse_res
        ]
        print(f"Retrieved {len(hits)} sparse documents.")
        return hits
