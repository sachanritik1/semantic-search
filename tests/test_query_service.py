from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.services.query_service import QueryService
from app.services.re_ranker import RerankResult
from app.services.semantic_cache import SemanticAskCache


@pytest.mark.asyncio
async def test_ask_uses_only_reranked_docs_on_success():
    fused = [
        Document(page_content="a", metadata={"chunk_id": "1", "fusion_score": 0.9}),
        Document(page_content="b", metadata={"chunk_id": "2", "fusion_score": 0.5}),
    ]
    selected = [
        Document(
            page_content="a",
            metadata={"chunk_id": "1", "rerank_score": 9},
        ),
    ]

    llm = MagicMock()
    llm.generate_text.return_value = MagicMock(content="answer")

    enhancer = MagicMock()
    enhancer.enhance.return_value = ["q1", "q2", "q3"]

    service = QueryService(llm_service=llm, query_enhancer=enhancer)

    with (
        patch(
            "app.services.query_service.list_chunks_for_document",
            return_value=[MagicMock()],
        ),
        patch.object(service, "_retrieve_dense", return_value=[]),
        patch.object(service, "_build_sparse_retriever", return_value=(MagicMock(), [])),
        patch.object(service, "_retrieve_sparse_with_index", return_value=[]),
        patch(
            "app.services.query_service.merge_hit_lists",
            side_effect=lambda hits: hits,
        ),
        patch(
            "app.services.query_service.fuse_documents",
            return_value=fused,
        ),
        patch(
            "app.services.query_service.re_rank_docs",
            return_value=RerankResult(docs=selected, failed=False),
        ),
        patch(
            "app.services.query_service.build_prompt",
            return_value="prompt",
        ) as build_prompt,
    ):
        result = await service.ask("question?", document_id="doc-1")

    assert result["response"] == "answer"
    assert result["enhanced_questions"] == ["q1", "q2", "q3"]
    assert result["enhanced_question"] == "q1 | q2 | q3"
    build_prompt.assert_called_once()
    assert build_prompt.call_args.kwargs["docs"] == selected


@pytest.mark.asyncio
async def test_ask_uses_all_fused_on_rerank_failure():
    fused = [
        Document(page_content="a", metadata={"chunk_id": "1"}),
        Document(page_content="b", metadata={"chunk_id": "2"}),
    ]

    llm = MagicMock()
    llm.generate_text.return_value = MagicMock(content="answer")

    enhancer = MagicMock()
    enhancer.enhance.return_value = ["q1", "q2", "q3"]

    service = QueryService(llm_service=llm, query_enhancer=enhancer)

    with (
        patch(
            "app.services.query_service.list_chunks_for_document",
            return_value=[MagicMock()],
        ),
        patch.object(service, "_retrieve_dense", return_value=[]),
        patch.object(service, "_build_sparse_retriever", return_value=(MagicMock(), [])),
        patch.object(service, "_retrieve_sparse_with_index", return_value=[]),
        patch(
            "app.services.query_service.merge_hit_lists",
            side_effect=lambda hits: hits,
        ),
        patch(
            "app.services.query_service.fuse_documents",
            return_value=fused,
        ),
        patch(
            "app.services.query_service.re_rank_docs",
            return_value=RerankResult(docs=[], failed=True),
        ),
        patch(
            "app.services.query_service.build_prompt",
            return_value="prompt",
        ) as build_prompt,
    ):
        await service.ask("question?", document_id="doc-1")

    assert build_prompt.call_args.kwargs["docs"] == fused


@pytest.mark.asyncio
async def test_ask_passes_each_query_to_retrievers():
    llm = MagicMock()
    llm.generate_text.return_value = MagicMock(content="answer")

    enhancer = MagicMock()
    enhancer.enhance.return_value = ["q1", "q2", "q3"]

    service = QueryService(llm_service=llm, query_enhancer=enhancer)
    document_id = "doc-123"
    sparse_bundle = (MagicMock(), [])

    with (
        patch.object(service, "_retrieve_dense", return_value=[]) as retrieve_dense,
        patch.object(
            service,
            "_build_sparse_retriever",
            return_value=sparse_bundle,
        ) as build_sparse,
        patch.object(
            service,
            "_retrieve_sparse_with_index",
            return_value=[],
        ) as retrieve_sparse,
        patch(
            "app.services.query_service.list_chunks_for_document",
            return_value=[MagicMock()],
        ),
        patch(
            "app.services.query_service.merge_hit_lists",
            side_effect=lambda hits: hits,
        ),
        patch(
            "app.services.query_service.fuse_documents",
            return_value=[],
        ),
        patch(
            "app.services.query_service.build_prompt",
            return_value="prompt",
        ),
    ):
        await service.ask("question?", document_id=document_id)

    build_sparse.assert_called_once_with(document_id)
    assert retrieve_dense.call_count == 3
    retrieve_dense.assert_any_call("q1", document_id=document_id)
    retrieve_dense.assert_any_call("q2", document_id=document_id)
    retrieve_dense.assert_any_call("q3", document_id=document_id)
    assert retrieve_sparse.call_count == 3
    retrieve_sparse.assert_any_call(
        sparse_bundle,
        "q1",
        document_id=document_id,
    )


@pytest.mark.asyncio
async def test_ask_returns_early_when_scoped_document_has_no_chunks():
    llm = MagicMock()
    llm.generate_text.return_value = MagicMock(content="no context answer")

    enhancer = MagicMock()
    enhancer.enhance.return_value = ["q1", "q2", "q3"]

    service = QueryService(llm_service=llm, query_enhancer=enhancer)

    with (
        patch(
            "app.services.query_service.list_chunks_for_document",
            return_value=[],
        ),
        patch.object(service, "_retrieve_dense") as retrieve_dense,
        patch.object(service, "_build_sparse_retriever") as build_sparse,
        patch(
            "app.services.query_service.build_prompt",
            return_value="prompt",
        ) as build_prompt,
    ):
        result = await service.ask("question?", document_id="missing-doc")

    retrieve_dense.assert_not_called()
    build_sparse.assert_not_called()
    build_prompt.assert_called_once_with(docs=[], question="question?")
    assert result["response"] == "no context answer"
    assert result["enhanced_questions"] == ["q1", "q2", "q3"]


@pytest.mark.asyncio
async def test_ask_returns_cache_hit_on_repeat_question(isolated_ask_cache_db):
    llm = MagicMock()
    llm.generate_text.return_value = MagicMock(content="cached answer")

    enhancer = MagicMock()
    enhancer.enhance.return_value = ["q1", "q2", "q3"]

    embeddings = MagicMock()
    embeddings.embed_query.return_value = [1.0, 0.0]
    cache = SemanticAskCache(embeddings, threshold=0.99, ttl_seconds=3600)

    service = QueryService(
        llm_service=llm,
        query_enhancer=enhancer,
        semantic_cache=cache,
    )

    with (
        patch(
            "app.services.query_service.list_chunks_for_document",
            return_value=[],
        ),
        patch(
            "app.services.query_service.build_prompt",
            return_value="prompt",
        ),
    ):
        first = await service.ask("question?", document_id="doc-1")
        second = await service.ask("question?", document_id="doc-1")

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["response"] == "cached answer"
    enhancer.enhance.assert_called_once()
    llm.generate_text.assert_called_once()
