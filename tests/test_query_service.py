from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.services.query_service import QueryService
from app.services.re_ranker import RerankResult


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
    enhancer.enhance.return_value = "enhanced q"

    service = QueryService(llm_service=llm, query_enhancer=enhancer)

    with (
        patch(
            "app.services.query_service.list_chunks_for_document",
            return_value=[MagicMock()],
        ),
        patch.object(service, "_retrieve_dense", return_value=[]),
        patch.object(service, "_retrieve_sparse", return_value=[]),
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
    enhancer.enhance.return_value = "q"

    service = QueryService(llm_service=llm, query_enhancer=enhancer)

    with (
        patch(
            "app.services.query_service.list_chunks_for_document",
            return_value=[MagicMock()],
        ),
        patch.object(service, "_retrieve_dense", return_value=[]),
        patch.object(service, "_retrieve_sparse", return_value=[]),
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
async def test_ask_passes_document_id_to_retrievers():
    llm = MagicMock()
    llm.generate_text.return_value = MagicMock(content="answer")

    enhancer = MagicMock()
    enhancer.enhance.return_value = "enhanced q"

    service = QueryService(llm_service=llm, query_enhancer=enhancer)
    document_id = "doc-123"

    with (
        patch.object(service, "_retrieve_dense", return_value=[]) as retrieve_dense,
        patch.object(service, "_retrieve_sparse", return_value=[]) as retrieve_sparse,
        patch(
            "app.services.query_service.list_chunks_for_document",
            return_value=[MagicMock()],
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

    retrieve_dense.assert_called_once_with(
        "enhanced q",
        document_id=document_id,
    )
    retrieve_sparse.assert_called_once_with(
        "enhanced q",
        document_id=document_id,
    )


@pytest.mark.asyncio
async def test_ask_returns_early_when_scoped_document_has_no_chunks():
    llm = MagicMock()
    llm.generate_text.return_value = MagicMock(content="no context answer")

    enhancer = MagicMock()
    enhancer.enhance.return_value = "enhanced q"

    service = QueryService(llm_service=llm, query_enhancer=enhancer)

    with (
        patch(
            "app.services.query_service.list_chunks_for_document",
            return_value=[],
        ),
        patch.object(service, "_retrieve_dense") as retrieve_dense,
        patch.object(service, "_retrieve_sparse") as retrieve_sparse,
        patch(
            "app.services.query_service.build_prompt",
            return_value="prompt",
        ) as build_prompt,
    ):
        result = await service.ask("question?", document_id="missing-doc")

    retrieve_dense.assert_not_called()
    retrieve_sparse.assert_not_called()
    build_prompt.assert_called_once_with(docs=[], question="enhanced q")
    assert result["response"] == "no context answer"
