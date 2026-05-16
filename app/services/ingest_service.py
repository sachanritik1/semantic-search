import os
import tempfile

from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader

from app.db.document_store import save_documents
from app.db.vector_store import upsert_documents
from app.services.embedder import embeddings
from app.utils.chunker import text_splitter
from app.utils.ids import new_document_id, stamp_document_chunks


class IngestService:
    async def ingest_pdf(self, file: UploadFile) -> dict:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        finally:
            os.unlink(tmp_path)

        all_splits = text_splitter.split_documents(docs)
        document_id = new_document_id()
        source = file.filename or "upload.pdf"
        stamp_document_chunks(all_splits, document_id=document_id, source=source)

        upsert_documents(embeddings, all_splits)
        saved = save_documents(all_splits)

        return {
            "document_id": document_id,
            "chunks_total": len(all_splits),
            "chunks_saved": saved,
        }
