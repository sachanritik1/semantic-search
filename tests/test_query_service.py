import asyncio
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.services.query_service import QueryService
from app.services.re_ranker import RerankResult
from app.services.semantic_cache import SemanticAskCache


@pytest.mark.asyncio
async def test_ask_uses_only_reranked_docs_on_success():
    fused = [
        Document(page_content="a", metadata={"chunk_id": "1"}),
        Document(page_content="b", metadata={"chunk_id": "2"}),
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
            "app.services.query_service.document_has_chunks",
            return_value=[MagicMock()],
        ),
        patch(
            "app.services.query_service.hybrid_search",
            return_value=[(fused[0], 0.9), (fused[1], 0.5)],
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
            "app.services.query_service.document_has_chunks",
            return_value=[MagicMock()],
        ),
        patch(
            "app.services.query_service.hybrid_search",
            return_value=[(fused[0], 0.9), (fused[1], 0.5)],
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
async def test_ask_passes_each_query_to_hybrid_search():
    llm = MagicMock()
    llm.generate_text.return_value = MagicMock(content="answer")

    enhancer = MagicMock()
    enhancer.enhance.return_value = ["q1", "q2", "q3"]

    service = QueryService(llm_service=llm, query_enhancer=enhancer)
    document_id = "doc-123"

    chunks = [MagicMock() for _ in range(3)]
    for i, c in enumerate(chunks):
        c.content = f"content {i}"

    fake_doc = Document(page_content="test", metadata={"chunk_id": "c1"})

    with (
        patch(
            "app.services.query_service.document_has_chunks",
            return_value=chunks,
        ),
        patch(
            "app.services.query_service.hybrid_search",
            return_value=[(fake_doc, 0.95)],
        ) as mock_hybrid,
        patch(
            "app.services.query_service.build_ask_messages",
            return_value=("system", "user"),
        ),
    ):
        await service.ask("question?", document_id=document_id)

    assert mock_hybrid.call_count == 3
    for call in mock_hybrid.call_args_list:
        assert call.kwargs["document_id"] == document_id


@pytest.mark.asyncio
async def test_ask_returns_early_when_scoped_document_has_no_chunks():
    llm = MagicMock()
    llm.generate_text.return_value = MagicMock(content="no context answer")

    enhancer = MagicMock()
    enhancer.enhance.return_value = ["q1", "q2", "q3"]

    service = QueryService(llm_service=llm, query_enhancer=enhancer)

    with (
        patch(
            "app.services.query_service.document_has_chunks",
            return_value=[],
        ),
        patch(
            "app.services.query_service.hybrid_search",
        ) as mock_hybrid,
        patch(
            "app.services.query_service.build_ask_messages",
            return_value=("system", "user"),
        ) as build_ask_messages,
    ):
        result = await service.ask("question?", document_id="missing-doc")

    mock_hybrid.assert_not_called()
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
            "app.services.query_service.document_has_chunks",
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
            "app.services.query_service.document_has_chunks",
            return_value=[],
        ),
        patch(
            "app.services.query_service.build_ask_messages",
            return_value=("system", "user"),
        ),
    ):
        gen = service.stream_ask("question?", document_id="doc-1")
        events: list[dict] = []
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

        for _ in range(20):
            await asyncio.sleep(0)

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
            "app.services.query_service.document_has_chunks",
            return_value=[MagicMock()],
        ),
        patch(
            "app.services.query_service.hybrid_search",
            return_value=[(fused[0], 0.9)],
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

    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_stream_ask_skips_reranking_stage_when_no_fused_docs():
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
            "app.services.query_service.document_has_chunks",
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
    import time

    from app.llm.base import BaseLLM, LLMResponse

    class _SlowSyncLLM(BaseLLM):
        def __init__(self) -> None:
            self.produced_at: list[float] = []

        def generate(
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

    assert len(received_at) == 3
    for produced, received in zip(llm.produced_at, received_at):
        assert received - produced < 0.04, (
            f"chunk received too long after production: {received - produced:.3f}s"
        )
