from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.dependencies import get_ingest_service
from app.services.ingest_service import IngestService

router = APIRouter(tags=["ingest"])


@router.post("/ingest")
async def ingest_data(
    file: UploadFile = File(...),
    ingest_service: IngestService = Depends(get_ingest_service),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    return await ingest_service.ingest_pdf(file_bytes=file_bytes, filename=file.filename)
