import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langchain_core.documents import Document
from langfuse import get_client, observe

from app.config import settings
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
from app.utils.prompt_cache import cache_key
from app.utils.prompts import build_ask_messages

logger = logging.getLogger(__name__)

# Background tasks that must outlive a cancelled SSE generator so the LLM
# call still completes and its result is persisted to the semantic cache.
# We hold strong references here to prevent garbage collection.
_pending_completions: set[asyncio.Task] = set()


@dataclass
class PrepareResult:
    result_base: dict
    system_prompt: str | None
    user_message: str | None
    cached_response: str | None
    cache_hit: bool


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

    def _cache_lookup(self, question: str, document_id: str) -> dict | None:
        if not self.semantic_cache:
            return None
        return self.semantic_cache.lookup(question, document_id)

    def _enhance_queries(self, question: str) -> tuple[list[str], dict]:
        queries = self.query_enhancer.enhance(question)
        if not queries:
            queries = [question]
        result_base = {
            "original_question": question,
            "enhanced_question": " | ".join(queries),
            "enhanced_questions": queries,
        }
        return queries, result_base

    def _retrieve_fused(
        self,
        queries: list[str],
        document_id: str,
    ) -> list[Document]:
        if not list_chunks_for_document(document_id):
            logger.info(
                "No chunks found for document_id=%s; answering without retrieval context",
                document_id,
            )
            return []

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
        return filter_fused_documents(
            fused,
            min_score=settings.FUSION_MIN_SCORE,
            min_docs=settings.FUSION_MIN_DOCS,
        )

    def _rerank(
        self,
        question: str,
        fused: list[Document],
    ) -> list[Document]:
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
            return fused
        return rerank_result.docs

    def _build_messages_for(
        self,
        question: str,
        queries: list[str],
        answer_docs: list[Document],
    ) -> tuple[str, str]:
        if not answer_docs:
            return build_ask_messages(docs=[], question=question)
        search_query = " | ".join(queries) if len(queries) > 1 else queries[0]
        return build_ask_messages(
            docs=answer_docs,
            question=question,
            search_query=search_query,
        )

    def _prepare_ask(self, question: str, *, document_id: str) -> PrepareResult:
        cached = self._cache_lookup(question, document_id)
        if cached is not None:
            return PrepareResult(
                result_base={
                    "original_question": cached["original_question"],
                    "enhanced_question": cached["enhanced_question"],
                },
                system_prompt=None,
                user_message=None,
                cached_response=cached["response"],
                cache_hit=True,
            )

        queries, result_base = self._enhance_queries(question)
        fused = self._retrieve_fused(queries, document_id)
        answer_docs = self._rerank(question, fused) if fused else []
        system_prompt, user_message = self._build_messages_for(
            question, queries, answer_docs
        )
        return PrepareResult(
            result_base=result_base,
            system_prompt=system_prompt,
            user_message=user_message,
            cached_response=None,
            cache_hit=False,
        )

    @observe(name="ask", capture_input=False)
    async def ask(self, question: str, *, document_id: str) -> dict:
        get_client().update_current_span(
            input={"question": question, "document_id": document_id},
        )
        prepared = await asyncio.to_thread(
            self._prepare_ask,
            question,
            document_id=document_id,
        )
        if prepared.cache_hit:
            result = {
                **prepared.result_base,
                "response": prepared.cached_response,
                "cache_hit": True,
            }
            get_client().update_current_span(output=prepared.cached_response)
            return result

        if prepared.user_message is None:
            raise RuntimeError("Prepared ask payload missing user message")

        response = self.llm_service.generate_text(
            prepared.user_message,
            system_prompt=prepared.system_prompt,
            cache_key=cache_key("ask"),
        )
        result = {
            **prepared.result_base,
            "response": response.content,
        }
        if response.usage:
            result["usage"] = response.usage
        result = self._complete_ask(question, document_id, result)
        get_client().update_current_span(output=result["response"])
        return result

    @observe(name="ask.stream", capture_input=False)
    async def stream_ask(
        self,
        question: str,
        *,
        document_id: str,
    ) -> AsyncIterator[dict]:
        get_client().update_current_span(
            input={"question": question, "document_id": document_id},
        )
        # Cache lookup is fast but still touches Postgres + numpy; run off the
        # event loop so the very first SSE write happens immediately and the
        # client can render "preparing" while we look.
        cached = await asyncio.to_thread(
            self._cache_lookup,
            question,
            document_id,
        )
        if cached is not None:
            meta = {
                "original_question": cached["original_question"],
                "enhanced_question": cached["enhanced_question"],
                "cache_hit": True,
            }
            yield {"event": "meta", "data": meta}
            get_client().update_current_span(output=cached["response"])
            yield {"event": "token", "data": {"text": cached["response"]}}
            yield {"event": "done", "data": {"cache_hit": True}}
            return

        # Each stage runs in a worker thread, with a status event emitted
        # before it starts so the UI can show what is happening right now.
        yield {"event": "status", "data": {"stage": "enhancing_query"}}
        queries, result_base = await asyncio.to_thread(self._enhance_queries, question)

        yield {"event": "status", "data": {"stage": "retrieving"}}
        fused = await asyncio.to_thread(self._retrieve_fused, queries, document_id)

        if fused:
            yield {"event": "status", "data": {"stage": "reranking"}}
            answer_docs = await asyncio.to_thread(self._rerank, question, fused)
        else:
            answer_docs = []

        system_prompt, user_message = await asyncio.to_thread(
            self._build_messages_for, question, queries, answer_docs
        )

        meta = {**result_base}
        yield {"event": "meta", "data": meta}
        yield {"event": "status", "data": {"stage": "generating"}}

        prepared = PrepareResult(
            result_base=result_base,
            system_prompt=system_prompt,
            user_message=user_message,
            cached_response=None,
            cache_hit=False,
        )

        # Detach LLM generation so it completes even if the SSE consumer
        # disconnects mid-stream. The bg task writes the full response to the
        # semantic cache; we forward chunks to the wire while connected.
        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()
        full_text: list[str] = []

        async def _drive_and_cache() -> None:
            try:
                if prepared.user_message is None:
                    raise RuntimeError("Prepared ask payload missing user message")
                async for chunk in self.llm_service.stream_text(
                    prepared.user_message,
                    system_prompt=prepared.system_prompt,
                    cache_key=cache_key("ask"),
                ):
                    full_text.append(chunk)
                    queue.put_nowait(chunk)
            except BaseException as exc:
                queue.put_nowait(exc)
                logger.exception(
                    "Detached LLM stream failed for document_id=%s", document_id
                )
                return
            finally:
                queue.put_nowait(_SENTINEL)

            try:
                self._complete_ask(
                    question,
                    document_id,
                    {
                        **prepared.result_base,
                        "response": "".join(full_text),
                    },
                )
            except Exception:
                logger.exception(
                    "Failed to persist completed ask to cache for document_id=%s",
                    document_id,
                )

        bg_task = asyncio.create_task(_drive_and_cache())
        _pending_completions.add(bg_task)
        bg_task.add_done_callback(_pending_completions.discard)

        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            yield {"event": "token", "data": {"text": item}}

        get_client().update_current_span(output="".join(full_text))
        yield {"event": "done", "data": {"cache_hit": False}}

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
        hits = [(chunk_to_document(chunks[idx]), score) for idx, score, _ in sparse_res]
        logger.debug(
            "Retrieved %d sparse documents for document_id=%s query=%r",
            len(hits),
            document_id,
            query,
        )
        return hits
