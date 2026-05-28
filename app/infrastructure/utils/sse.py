import asyncio
import json
from collections.abc import AsyncIterator

SSE_PING_FRAME = ": ping\n\n"


def format_sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def sse_from_events(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for item in events:
        yield format_sse_event(item["event"], item["data"])


async def with_heartbeats(
    frames: AsyncIterator[str],
    *,
    interval_s: float,
) -> AsyncIterator[str]:
    """Merge SSE frames with comment heartbeats on a fixed interval.

    If the upstream `frames` generator raises, the exception is re-raised on
    the consumer side so callers can translate it into an SSE `error` event
    instead of silently terminating the stream.
    """
    queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()
    done = asyncio.Event()

    async def pump_frames() -> None:
        try:
            async for frame in frames:
                await queue.put(frame)
        except BaseException as exc:
            await queue.put(exc)
        finally:
            done.set()
            await queue.put(None)

    async def pump_heartbeats() -> None:
        try:
            while not done.is_set():
                await asyncio.sleep(interval_s)
                if not done.is_set():
                    await queue.put(SSE_PING_FRAME)
        except asyncio.CancelledError:
            pass

    frames_task = asyncio.create_task(pump_frames())
    hb_task = asyncio.create_task(pump_heartbeats())
    try:
        while True:
            item = await queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        done.set()
        hb_task.cancel()
        frames_task.cancel()
        await asyncio.gather(hb_task, frames_task, return_exceptions=True)
