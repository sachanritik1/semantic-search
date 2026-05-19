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


def retrieval_methods_for_key(
    key: str,
    dense_raw: dict[str, float],
    sparse_raw: dict[str, float],
) -> list[str]:
    """Return retrieval channels that returned this chunk (dense, sparse, or both)."""
    methods: list[str] = []
    if key in dense_raw:
        methods.append("dense")
    if key in sparse_raw:
        methods.append("sparse")
    return methods


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {key: 1.0 for key in scores}
    span = hi - lo
    return {key: (value - lo) / span for key, value in scores.items()}


def merge_hit_lists(
    hits: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """Dedupe hits by chunk_key, keeping the maximum raw score per chunk."""
    best: dict[str, tuple[Document, float]] = {}
    for doc, score in hits:
        key = chunk_key(doc)
        if key not in best or score > best[key][1]:
            best[key] = (doc, score)
    return list(best.values())


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
        metadata["retrieval_methods"] = retrieval_methods_for_key(
            key, dense_raw, sparse_raw
        )
        fused.append(Document(page_content=base.page_content, metadata=metadata))

    fused.sort(
        key=lambda doc: float((doc.metadata or {}).get("fusion_score", 0.0)),
        reverse=True,
    )
    return fused


def fusion_score(doc: Document) -> float:
    return float((doc.metadata or {}).get("fusion_score", 0.0))


def filter_fused_documents(
    fused: list[Document],
    *,
    min_score: float,
    min_docs: int = 10,
) -> list[Document]:
    """Drop low fusion-score docs, but keep at least min(min_docs, len(fused)) by score."""
    if not fused:
        return []

    floor = min(min_docs, len(fused))
    qualified = [doc for doc in fused if fusion_score(doc) >= min_score]

    if len(qualified) >= floor:
        return qualified

    return fused[:floor]


__all__ = [
    "chunk_key",
    "filter_fused_documents",
    "fuse_documents",
    "fusion_score",
    "merge_hit_lists",
    "min_max_normalize",
    "retrieval_methods_for_key",
]
