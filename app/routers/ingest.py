from fastapi import APIRouter, Depends, File, UploadFile

from app.dependencies import get_ingest_service
from app.services.ingest_service import IngestService

router = APIRouter(tags=["ingest"])


@router.post("/ingest")
async def ingest_data(
    file: UploadFile = File(...),
    ingest_service: IngestService = Depends(get_ingest_service),
):
    return await ingest_service.ingest_pdf(file)
