import asyncio
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.dependencies import get_query_service
from app.main import app

client = TestClient(app)


def _parse_sse_events(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_line = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_line = line[5:].strip()
        events.append((event_name, json.loads(data_line)))
    return events


def test_ask_requires_document_id():
    response = client.post("/ask", json={"question": "What is this about?"})
    assert response.status_code == 422


def test_ask_rejects_empty_document_id():
    response = client.post(
        "/ask",
        json={"question": "What is this about?", "document_id": ""},
    )
    assert response.status_code == 422


async def _mock_stream_ask(question: str, *, document_id: str):
    yield {
        "event": "meta",
        "data": {
            "original_question": question,
            "enhanced_question": question,
            "enhanced_questions": [question],
        },
    }
    yield {"event": "token", "data": {"text": "Hel"}}
    yield {"event": "token", "data": {"text": "lo"}}
    yield {"event": "done", "data": {"cache_hit": False}}


def test_ask_stream_returns_sse_events():
    mock_service = MagicMock()
    mock_service.stream_ask = _mock_stream_ask
    app.dependency_overrides[get_query_service] = lambda: mock_service
    try:
        response = client.post(
            "/ask/stream",
            json={"question": "What is this about?", "document_id": "doc-1"},
        )
    finally:
        app.dependency_overrides.pop(get_query_service, None)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    assert [name for name, _ in events] == ["meta", "token", "token", "done"]
    assert events[0][1]["original_question"] == "What is this about?"
    assert events[1][1]["text"] == "Hel"
    assert events[2][1]["text"] == "lo"
    assert events[3][1]["cache_hit"] is False


def test_ask_stream_requires_document_id():
    response = client.post("/ask/stream", json={"question": "What is this about?"})
    assert response.status_code == 422


async def _slow_stream_ask(question: str, *, document_id: str):
    yield {
        "event": "meta",
        "data": {
            "original_question": question,
            "enhanced_question": question,
        },
    }
    await asyncio.sleep(0.05)
    yield {"event": "done", "data": {"cache_hit": False}}


def test_ask_stream_emits_heartbeats(monkeypatch):
    monkeypatch.setattr(settings, "SSE_HEARTBEAT_INTERVAL_S", 0.02)
    mock_service = MagicMock()
    mock_service.stream_ask = _slow_stream_ask
    app.dependency_overrides[get_query_service] = lambda: mock_service
    try:
        response = client.post(
            "/ask/stream",
            json={"question": "What is this about?", "document_id": "doc-1"},
        )
    finally:
        app.dependency_overrides.pop(get_query_service, None)

    assert response.status_code == 200
    assert ": ping" in response.text

