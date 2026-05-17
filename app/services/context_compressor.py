from copy import deepcopy

from langchain_core.documents import Document

from app.config import settings


def max_chars_for_doc(doc: Document, rank: int) -> int:
    meta = doc.metadata or {}
    rerank_score = meta.get("rerank_score")
    score = float(rerank_score) if rerank_score is not None else 0.0

    if score >= 8 or rank <= 2:
        return settings.CONTEXT_MAX_CHARS_HIGH
    if score >= 5:
        return settings.CONTEXT_MAX_CHARS_MID
    return settings.CONTEXT_MAX_CHARS_LOW


def compress_documents_for_context(docs: list[Document]) -> list[Document]:
    """Return copies with tiered truncation for the answer prompt."""
    compressed: list[Document] = []
    for rank, doc in enumerate(docs, start=1):
        limit = max_chars_for_doc(doc, rank)
        content = doc.page_content
        if len(content) > limit:
            content = content[:limit] + "…"
        compressed.append(
            Document(page_content=content, metadata=deepcopy(doc.metadata or {}))
        )
    return compressed


__all__ = ["compress_documents_for_context", "max_chars_for_doc"]
