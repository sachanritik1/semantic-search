import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.config import settings
from app.dependencies import get_query_service
from app.schemas.common import AskRequest
from app.services.query_service import QueryService
from app.utils.sse import format_sse_event, sse_from_events, with_heartbeats

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.post("/ask")
async def ask_question(
    request: AskRequest,
    query_service: QueryService = Depends(get_query_service),
):
    return await query_service.ask(
        request.question,
        document_id=request.document_id,
    )


async def _ask_stream_generator(
    query_service: QueryService,
    question: str,
    document_id: str,
):
    try:
        frames = sse_from_events(
            query_service.stream_ask(question, document_id=document_id),
        )
        async for frame in with_heartbeats(
            frames,
            interval_s=settings.SSE_HEARTBEAT_INTERVAL_S,
        ):
            yield frame
    except Exception as exc:
        logger.exception("ask/stream failed")
        yield format_sse_event("error", {"message": str(exc)})


@router.post("/ask/stream")
async def ask_question_stream(
    request: AskRequest,
    query_service: QueryService = Depends(get_query_service),
):
    return StreamingResponse(
        _ask_stream_generator(
            query_service,
            request.question,
            request.document_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
