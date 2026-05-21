import asyncio

import pytest

from app.utils.sse import SSE_PING_FRAME, format_sse_event, with_heartbeats


@pytest.mark.asyncio
async def test_with_heartbeats_emits_ping_between_slow_frames():
    async def slow_frames():
        yield format_sse_event("meta", {"q": "x"})
        await asyncio.sleep(0.05)
        yield format_sse_event("done", {"cache_hit": False})

    output: list[str] = []
    async for frame in with_heartbeats(slow_frames(), interval_s=0.02):
        output.append(frame)

    assert any(frame == SSE_PING_FRAME for frame in output)
    assert any("event: meta" in frame for frame in output)
    assert any("event: done" in frame for frame in output)
