from unittest.mock import AsyncMock, MagicMock, patch

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
        patch.object(service, "_retrieve_dense", return_value=[]),
        patch.object(service, "_retrieve_sparse", return_value=[]),
        patch(
            "app.services.query_service.fuse_documents",
            return_value=fused,
        ),
        patch(
            "app.services.query_service.re_rank_docs",
            new_callable=AsyncMock,
            return_value=RerankResult(docs=selected, failed=False),
        ),
        patch(
            "app.services.query_service.compress_documents_for_context",
            side_effect=lambda docs: docs,
        ) as compress,
        patch(
            "app.services.query_service.build_prompt",
            return_value="prompt",
        ) as build_prompt,
    ):
        result = await service.ask("question?")

    assert result["response"] == "answer"
    compress.assert_called_once_with(selected)
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
        patch.object(service, "_retrieve_dense", return_value=[]),
        patch.object(service, "_retrieve_sparse", return_value=[]),
        patch(
            "app.services.query_service.fuse_documents",
            return_value=fused,
        ),
        patch(
            "app.services.query_service.re_rank_docs",
            new_callable=AsyncMock,
            return_value=RerankResult(docs=[], failed=True),
        ),
        patch(
            "app.services.query_service.compress_documents_for_context",
            side_effect=lambda docs: docs,
        ) as compress,
        patch(
            "app.services.query_service.build_prompt",
            return_value="prompt",
        ),
    ):
        await service.ask("question?")

    compress.assert_called_once_with(fused)
