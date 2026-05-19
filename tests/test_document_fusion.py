import pytest
from langchain_core.documents import Document

from app.services.document_fusion import (
    chunk_key,
    filter_fused_documents,
    fuse_documents,
    merge_hit_lists,
    min_max_normalize,
)


def test_chunk_key_prefers_chunk_id():
    doc = Document(page_content="text", metadata={"chunk_id": "abc"})
    assert chunk_key(doc) == "abc"


def test_min_max_normalize_single_value():
    assert min_max_normalize({"a": 3.0}) == {"a": 1.0}


def test_min_max_normalize_range():
    assert min_max_normalize({"a": 0.0, "b": 10.0}) == {"a": 0.0, "b": 1.0}


def test_merge_hit_lists_keeps_max_score_per_chunk():
    hits = [
        (Document(page_content="shared", metadata={"chunk_id": "c1"}), 0.4),
        (Document(page_content="shared", metadata={"chunk_id": "c1"}), 0.9),
        (Document(page_content="other", metadata={"chunk_id": "c2"}), 0.5),
    ]
    merged = merge_hit_lists(hits)
    assert len(merged) == 2
    by_key = {chunk_key(doc): score for doc, score in merged}
    assert by_key["c1"] == 0.9
    assert by_key["c2"] == 0.5


def test_fuse_documents_dedupes_by_chunk_id():
    dense = [
        (
            Document(page_content="shared", metadata={"chunk_id": "c1"}),
            0.9,
        ),
        (
            Document(page_content="dense-only", metadata={"chunk_id": "c2"}),
            0.5,
        ),
    ]
    sparse = [
        (
            Document(page_content="shared", metadata={"chunk_id": "c1"}),
            8.0,
        ),
        (
            Document(page_content="sparse-only", metadata={"chunk_id": "c3"}),
            4.0,
        ),
    ]

    fused = fuse_documents(dense, sparse, dense_weight=0.5, sparse_weight=0.5)
    keys = {doc.metadata["chunk_id"] for doc in fused}
    assert keys == {"c1", "c2", "c3"}
    assert fused[0].metadata["chunk_id"] == "c1"
    assert fused[0].metadata["fusion_score"] > 0

    by_id = {doc.metadata["chunk_id"]: doc.metadata for doc in fused}
    assert by_id["c1"]["retrieval_methods"] == ["dense", "sparse"]
    assert by_id["c2"]["retrieval_methods"] == ["dense"]
    assert by_id["c3"]["retrieval_methods"] == ["sparse"]


def test_fuse_documents_single_channel_gets_partial_score():
    dense = [
        (Document(page_content="only dense", metadata={"chunk_id": "c1"}), 0.8),
    ]
    fused = fuse_documents(dense, [], dense_weight=0.6, sparse_weight=0.4)
    assert len(fused) == 1
    meta = fused[0].metadata
    assert meta["dense_norm"] == 1.0
    assert meta["sparse_norm"] == 0.0
    assert meta["fusion_score"] == pytest.approx(0.6)
    assert meta["retrieval_methods"] == ["dense"]


def _fused_doc(chunk_id: str, score: float) -> Document:
    return Document(
        page_content=chunk_id,
        metadata={"chunk_id": chunk_id, "fusion_score": score},
    )


def test_filter_fused_documents_drops_below_threshold():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.32, 0.31, 0.1, 0.05]
    fused = [_fused_doc(f"c{i}", s) for i, s in enumerate(scores)]
    filtered = filter_fused_documents(fused, min_score=0.3, min_docs=10)
    assert len(filtered) == 10
    assert all(d.metadata["fusion_score"] >= 0.3 for d in filtered)
    assert "c10" not in [d.metadata["chunk_id"] for d in filtered]


def test_filter_fused_documents_keeps_floor_when_too_few_qualify():
    fused = [_fused_doc(f"c{i}", score) for i, score in enumerate([0.9, 0.1, 0.05, 0.02])]
    filtered = filter_fused_documents(fused, min_score=0.5, min_docs=3)
    assert len(filtered) == 3
    assert [d.metadata["chunk_id"] for d in filtered] == ["c0", "c1", "c2"]


def test_filter_fused_documents_floor_capped_by_total():
    fused = [_fused_doc("c0", 0.9), _fused_doc("c1", 0.1)]
    filtered = filter_fused_documents(fused, min_score=0.5, min_docs=10)
    assert len(filtered) == 2
