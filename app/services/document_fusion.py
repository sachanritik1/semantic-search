from __future__ import annotations

from copy import deepcopy

from langchain_core.documents import Document


def chunk_key(doc: Document) -> str:
    meta = doc.metadata or {}
    if meta.get("chunk_id"):
        return str(meta["chunk_id"])
    if meta.get("content_hash"):
        return str(meta["content_hash"])
    return doc.page_content.strip()


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {key: 1.0 for key in scores}
    span = hi - lo
    return {key: (value - lo) / span for key, value in scores.items()}


def fuse_documents(
    dense_hits: list[tuple[Document, float]],
    sparse_hits: list[tuple[Document, float]],
    *,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[Document]:
    """Merge dense and sparse hits with weighted normalized scores."""
    docs_by_key: dict[str, Document] = {}
    dense_raw: dict[str, float] = {}
    sparse_raw: dict[str, float] = {}

    for doc, score in dense_hits:
        key = chunk_key(doc)
        dense_raw[key] = score
        docs_by_key[key] = doc

    for doc, score in sparse_hits:
        key = chunk_key(doc)
        sparse_raw[key] = score
        if key not in docs_by_key:
            docs_by_key[key] = doc
        else:
            merged_meta = dict(docs_by_key[key].metadata or {})
            merged_meta.update(doc.metadata or {})
            docs_by_key[key] = Document(
                page_content=docs_by_key[key].page_content,
                metadata=merged_meta,
            )

    dense_norm = min_max_normalize(dense_raw)
    sparse_norm = min_max_normalize(sparse_raw)
    all_keys = set(dense_raw) | set(sparse_raw)

    fused: list[Document] = []
    for key in all_keys:
        dn = dense_norm.get(key, 0.0)
        sn = sparse_norm.get(key, 0.0)
        fusion_score = dense_weight * dn + sparse_weight * sn
        base = docs_by_key[key]
        metadata = deepcopy(base.metadata or {})
        if key in dense_raw:
            metadata["dense_score"] = dense_raw[key]
        metadata["dense_norm"] = dn
        if key in sparse_raw:
            metadata["sparse_score"] = sparse_raw[key]
        metadata["sparse_norm"] = sn
        metadata["fusion_score"] = fusion_score
        fused.append(Document(page_content=base.page_content, metadata=metadata))

    fused.sort(
        key=lambda doc: float((doc.metadata or {}).get("fusion_score", 0.0)),
        reverse=True,
    )
    return fused


__all__ = ["chunk_key", "fuse_documents", "min_max_normalize"]
