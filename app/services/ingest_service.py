import logging
import os
import tempfile
from typing import Any, TypedDict, cast

from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document
from langfuse import get_client
from langfuse import observe as langfuse_observe  # type: ignore[reportUnknownVariableType]

from app.db.document_store import save_documents
from app.db.weaviate_store import ensure_collection, upsert_documents
from app.services.document_processor import DocumentProcessor
from app.services.embedder import get_embeddings
from app.utils.chunker import text_splitter
from app.utils.ids import new_document_id, stamp_document_chunks

logger = logging.getLogger(__name__)

_CHUNK_PREVIEW_LEN = 200


class IngestResponse(TypedDict):
    document_id: str
    chunks_total: int
    chunks_saved: int


class IngestService:
    @langfuse_observe(name="ingest.pdf", capture_input=False)
    async def ingest_pdf(self, file: UploadFile) -> IngestResponse:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        file_bytes = await file.read()
        get_client().update_current_span(
            input={
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(file_bytes),
            }
        )

        with get_client().start_as_current_observation(
            name="parse_and_clean",
            input={"filename": file.filename},
        ) as span:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            try:
                processor = DocumentProcessor(tmp_path)
                cleaned_text = processor.clean()
                source = file.filename or "upload.pdf"
                docs = [Document(page_content=cleaned_text, metadata={"source": source})]
            finally:
                os.unlink(tmp_path)

            span.update(
                output={
                    "source": source,
                    "raw_length_chars": len(cleaned_text),
                    "line_count": cleaned_text.count("\n") + 1,
                }
            )

        with get_client().start_as_current_observation(
            name="chunk",
            input={"raw_length_chars": len(cleaned_text)},
        ) as span:
            all_splits = text_splitter.split_documents(docs)
            document_id = new_document_id()
            stamp_document_chunks(all_splits, document_id=document_id, source=source)

            chunks_preview: list[dict[str, object]] = []
            for i, chunk in enumerate(all_splits):
                meta = cast(dict[str, Any], dict(chunk.metadata)) if chunk.metadata else {}
                chunks_preview.append(
                    {
                        "chunk_index": i,
                        "chunk_id": meta.get("chunk_id", ""),
                        "character_count": len(chunk.page_content),
                        "preview": chunk.page_content[:_CHUNK_PREVIEW_LEN],
                    }
                )

            span.update(
                output={
                    "chunks_total": len(all_splits),
                    "chunk_size": text_splitter._chunk_size,  # type: ignore[reportPrivateUsage]
                    "chunk_overlap": text_splitter._chunk_overlap,  # type: ignore[reportPrivateUsage]
                    "chunks": chunks_preview,
                }
            )

        with get_client().start_as_current_observation(
            name="embed_and_upsert",
            input={"chunk_count": len(all_splits)},
        ) as span:
            embedder = get_embeddings()
            embeddings = embedder.embed_documents(
                [c.page_content for c in all_splits]
            )

            ensure_collection()
            upsert_documents(embeddings, all_splits)

            span.update(output={"vector_store": "weaviate", "embedded_count": len(all_splits)})

        with get_client().start_as_current_observation(
            name="save_metadata",
            input={"chunk_count": len(all_splits)},
        ) as span:
            saved = save_documents(all_splits)
            span.update(output={"saved_count": saved})

        result: IngestResponse = {
            "document_id": document_id,
            "chunks_total": len(all_splits),
            "chunks_saved": saved,
        }
        get_client().update_current_span(
            output={
                **result,
                "parsed_length_chars": len(cleaned_text),
                "chunk_preview_count": min(len(all_splits), len(chunks_preview)),
            }
        )
        return result
