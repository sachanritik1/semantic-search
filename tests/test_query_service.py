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
        patch.object(
            service, "_build_sparse_retriever", return_value=(MagicMock(), [])
        ),
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
            "app.services.query_service.build_ask_messages",
            return_value=("system", "user"),
        ) as build_ask_messages,
    ):
        result = await service.ask("question?", document_id="doc-1")

    assert result["response"] == "answer"
    assert result["enhanced_questions"] == ["q1", "q2", "q3"]
    assert result["enhanced_question"] == "q1 | q2 | q3"
    build_ask_messages.assert_called_once()
    assert build_ask_messages.call_args.kwargs["docs"] == selected
    llm.generate_text.assert_called_once_with(
        "user",
        system_prompt="system",
        cache_key="ask:v1",
    )


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
        patch.object(
            service, "_build_sparse_retriever", return_value=(MagicMock(), [])
        ),
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
            "app.services.query_service.build_ask_messages",
            return_value=("system", "user"),
        ) as build_ask_messages,
    ):
        await service.ask("question?", document_id="doc-1")

    assert build_ask_messages.call_args.kwargs["docs"] == fused


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
            "app.services.query_service.build_ask_messages",
            return_value=("system", "user"),
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
            "app.services.query_service.build_ask_messages",
            return_value=("system", "user"),
        ) as build_ask_messages,
    ):
        result = await service.ask("question?", document_id="missing-doc")

    retrieve_dense.assert_not_called()
    build_sparse.assert_not_called()
    build_ask_messages.assert_called_once_with(docs=[], question="question?")
    assert result["response"] == "no context answer"
    assert result["enhanced_questions"] == ["q1", "q2", "q3"]


@pytest.mark.skip(reason="DB-layer tests pending Postgres rewrite")
@pytest.mark.asyncio
async def test_ask_returns_cache_hit_on_repeat_question():
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
            "app.services.query_service.build_ask_messages",
            return_value=("system", "user"),
        ),
    ):
        first = await service.ask("question?", document_id="doc-1")
        second = await service.ask("question?", document_id="doc-1")

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["response"] == "cached answer"
    enhancer.enhance.assert_called_once()
    llm.generate_text.assert_called_once()


@pytest.mark.skip(reason="DB-layer tests pending Postgres rewrite")
@pytest.mark.asyncio
async def test_stream_ask_persists_cache_when_consumer_disconnects_mid_stream():
    """If the SSE consumer stops iterating early (client disconnect), the LLM
    call must still run to completion and the full response must be written to
    the semantic cache. The next identical question should then be a cache hit.
    """

    import asyncio

    llm = MagicMock()

    async def fake_stream_text(prompt: str, **kwargs):
        for piece in ["Hel", "lo ", "world"]:
            await asyncio.sleep(0)
            yield piece

    llm.stream_text = fake_stream_text

    enhancer = MagicMock()
    enhancer.enhance.return_value = ["q"]

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
            "app.services.query_service.build_ask_messages",
            return_value=("system", "user"),
        ),
    ):
        gen = service.stream_ask("question?", document_id="doc-1")
        events: list[dict] = []
        # Drain status/meta frames until the first token arrives, then
        # disconnect mid-stream and verify the bg task still completes.
        meta_event: dict | None = None
        first_token: dict | None = None
        async for ev in gen:
            events.append(ev)
            if ev["event"] == "meta":
                meta_event = ev
            elif ev["event"] == "token":
                first_token = ev
                break
        await gen.aclose()

        # Let the detached bg task complete (LLM finish + cache write).
        for _ in range(20):
            await asyncio.sleep(0)

        # Cache must now contain the FULL response, not just the first chunk.
        replay: list[dict] = []
        async for ev in service.stream_ask("question?", document_id="doc-1"):
            replay.append(ev)

    assert meta_event is not None
    assert first_token is not None
    assert first_token["event"] == "token"
    cached_token = next(ev for ev in replay if ev["event"] == "token")
    assert cached_token["data"]["text"] == "Hello world"
    assert any(ev["event"] == "done" and ev["data"]["cache_hit"] for ev in replay)


@pytest.mark.asyncio
async def test_stream_ask_emits_stage_status_events_in_order():
    """The streaming endpoint should announce each pipeline stage before it
    starts so the UI can render a progress indicator that updates in real time.
    """

    import asyncio

    llm = MagicMock()

    async def fake_stream_text(prompt: str, **kwargs):
        yield "ok"

    llm.stream_text = fake_stream_text

    enhancer = MagicMock()
    enhancer.enhance.return_value = ["q1", "q2"]

    service = QueryService(llm_service=llm, query_enhancer=enhancer)

    fused = [Document(page_content="a", metadata={"chunk_id": "1"})]
    selected = [
        Document(page_content="a", metadata={"chunk_id": "1", "rerank_score": 9}),
    ]

    with (
        patch(
            "app.services.query_service.list_chunks_for_document",
            return_value=[MagicMock()],
        ),
        patch.object(service, "_retrieve_dense", return_value=[]),
        patch.object(
            service, "_build_sparse_retriever", return_value=(MagicMock(), [])
        ),
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
            "app.services.query_service.filter_fused_documents",
            return_value=fused,
        ),
        patch(
            "app.services.query_service.re_rank_docs",
            return_value=RerankResult(docs=selected, failed=False),
        ),
        patch(
            "app.services.query_service.build_ask_messages",
            return_value=("system", "user"),
        ),
    ):
        events: list[dict] = []
        async for ev in service.stream_ask("question?", document_id="doc-1"):
            events.append(ev)

    stages = [ev["data"]["stage"] for ev in events if ev["event"] == "status"]
    assert stages == [
        "enhancing_query",
        "retrieving",
        "reranking",
        "generating",
    ]

    event_types = [ev["event"] for ev in events]
    assert event_types.index("meta") < event_types.index("token")
    assert "done" in event_types

    # Let any detached background completion task finish before exiting so the
    # tests don't leak warnings about pending tasks.
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_stream_ask_skips_reranking_stage_when_no_fused_docs():
    """If retrieval returns nothing, we should not announce a reranking stage."""

    import asyncio

    llm = MagicMock()

    async def fake_stream_text(prompt: str, **kwargs):
        yield "ok"

    llm.stream_text = fake_stream_text

    enhancer = MagicMock()
    enhancer.enhance.return_value = ["q1"]

    service = QueryService(llm_service=llm, query_enhancer=enhancer)

    with (
        patch(
            "app.services.query_service.list_chunks_for_document",
            return_value=[],
        ),
        patch(
            "app.services.query_service.build_ask_messages",
            return_value=("system", "user"),
        ),
    ):
        events: list[dict] = []
        async for ev in service.stream_ask("question?", document_id="doc-1"):
            events.append(ev)

    stages = [ev["data"]["stage"] for ev in events if ev["event"] == "status"]
    assert stages == ["enhancing_query", "retrieving", "generating"]

    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_stream_generate_async_yields_chunks_incrementally():
    """Regression: the default async wrapper must NOT await the executor task
    (which would block until the entire sync stream completes before yielding).
    First chunk must arrive before subsequent chunks are produced.
    """

    import time

    from app.llm.base import BaseLLM, LLMResponse

    class _SlowSyncLLM(BaseLLM):
        def __init__(self) -> None:
            self.produced_at: list[float] = []

        def generate(  # pragma: no cover
            self,
            prompt,
            *,
            temperature=0.7,
            max_tokens=None,
            model=None,
            system_prompt=None,
            cache_key=None,
        ):
            return LLMResponse(content="")

        def stream_generate(
            self,
            prompt,
            *,
            temperature=0.7,
            max_tokens=None,
            model=None,
            system_prompt=None,
            cache_key=None,
        ):
            for piece in ["a", "b", "c"]:
                time.sleep(0.05)
                self.produced_at.append(time.monotonic())
                yield piece

    llm = _SlowSyncLLM()
    received_at: list[float] = []
    async for chunk in llm.stream_generate_async("prompt"):
        received_at.append(time.monotonic())

    # If the wrapper awaits the executor, all received_at timestamps come
    # tightly together AFTER all produced_at timestamps. With true streaming,
    # each received_at[i] should be near produced_at[i].
    assert len(received_at) == 3
    for produced, received in zip(llm.produced_at, received_at):
        assert received - produced < 0.04, (
            f"chunk received too long after production: {received - produced:.3f}s"
        )
