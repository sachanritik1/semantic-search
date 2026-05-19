import logging

from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)
from app.db.document_store import chunk_to_document, list_chunks_for_document
from app.services.dense_retriever import DenseRetriever
from app.services.document_fusion import (
    filter_fused_documents,
    fuse_documents,
    merge_hit_lists,
)
from app.services.embedder import embeddings
from app.services.llm_service import LLMService
from app.services.query_enhancer import QueryEnhancer
from app.services.re_ranker import re_rank_docs
from app.services.semantic_cache import SemanticAskCache
from app.services.sparse_retriever import SparseRetriever
from app.utils.prompts import build_prompt


class QueryService:
    def __init__(
        self,
        llm_service: LLMService,
        query_enhancer: QueryEnhancer,
        semantic_cache: SemanticAskCache | None = None,
    ):
        self.llm_service = llm_service
        self.query_enhancer = query_enhancer
        self.semantic_cache = semantic_cache

    async def ask(self, question: str, *, document_id: str) -> dict:
        if self.semantic_cache:
            cached = self.semantic_cache.lookup(question, document_id)
            if cached is not None:
                return {**cached, "cache_hit": True}

        queries = self.query_enhancer.enhance(question)
        if not queries:
            queries = [question]

        enhanced_joined = " | ".join(queries)
        result_base = {
            "original_question": question,
            "enhanced_question": enhanced_joined,
            "enhanced_questions": queries,
        }

        if not list_chunks_for_document(document_id):
            logger.info(
                "No chunks found for document_id=%s; answering without retrieval context",
                document_id,
            )
            prompt_text = build_prompt(docs=[], question=question)
            response = self.llm_service.generate_text(prompt_text)
            return self._complete_ask(
                question,
                document_id,
                {
                    **result_base,
                    "response": response.content,
                },
            )

        all_dense: list[tuple[Document, float]] = []
        all_sparse: list[tuple[Document, float]] = []
        sparse = self._build_sparse_retriever(document_id)

        for q in queries:
            all_dense.extend(self._retrieve_dense(q, document_id=document_id))
            if sparse is not None:
                all_sparse.extend(
                    self._retrieve_sparse_with_index(
                        sparse,
                        q,
                        document_id=document_id,
                    )
                )

        dense_hits = merge_hit_lists(all_dense)
        sparse_hits = merge_hit_lists(all_sparse)

        fused = fuse_documents(
            dense_hits,
            sparse_hits,
            dense_weight=settings.DENSE_WEIGHT,
            sparse_weight=settings.SPARSE_WEIGHT,
        )
        fused = filter_fused_documents(
            fused,
            min_score=settings.FUSION_MIN_SCORE,
            min_docs=settings.FUSION_MIN_DOCS,
        )

        if not fused:
            prompt_text = build_prompt(docs=[], question=question)
            response = self.llm_service.generate_text(prompt_text)
            return self._complete_ask(
                question,
                document_id,
                {
                    **result_base,
                    "response": response.content,
                },
            )

        rerank_result = re_rank_docs(
            question,
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

        search_query = enhanced_joined if len(queries) > 1 else queries[0]
        prompt_text = build_prompt(
            docs=answer_docs,
            question=question,
            search_query=search_query,
        )
        response = self.llm_service.generate_text(prompt_text)

        return self._complete_ask(
            question,
            document_id,
            {
                **result_base,
                "response": response.content,
            },
        )

    def _complete_ask(
        self,
        question: str,
        document_id: str,
        result: dict,
    ) -> dict:
        if self.semantic_cache:
            self.semantic_cache.store(question, document_id, result)
        return {**result, "cache_hit": False}

    def _retrieve_dense(
        self,
        query: str,
        *,
        document_id: str,
    ) -> list[tuple[Document, float]]:
        dense = DenseRetriever(embeddings, default_k=settings.RETRIEVAL_TOP_K)
        hits = dense.retrieve_with_scores(
            query,
            k=settings.RETRIEVAL_TOP_K,
            document_id=document_id,
        )
        logger.debug("Retrieved %d dense documents for query=%r", len(hits), query)
        return hits

    def _build_sparse_retriever(
        self,
        document_id: str,
    ) -> tuple[SparseRetriever, list] | None:
        chunks = list_chunks_for_document(document_id)
        if not chunks:
            return None

        texts = [c.content for c in chunks]
        sparse = SparseRetriever()
        sparse.build_index(texts)
        return sparse, chunks

    def _retrieve_sparse_with_index(
        self,
        sparse_and_chunks: tuple[SparseRetriever, list],
        query: str,
        *,
        document_id: str,
    ) -> list[tuple[Document, float]]:
        sparse, chunks = sparse_and_chunks
        sparse_res = sparse.query(query, top_k=settings.RETRIEVAL_TOP_K)
        hits = [
            (chunk_to_document(chunks[idx]), score) for idx, score, _ in sparse_res
        ]
        logger.debug(
            "Retrieved %d sparse documents for document_id=%s query=%r",
            len(hits),
            document_id,
            query,
        )
        return hits
