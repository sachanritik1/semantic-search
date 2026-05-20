import json
from collections.abc import AsyncIterator


def format_sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def sse_from_events(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for item in events:
        yield format_sse_event(item["event"], item["data"])
