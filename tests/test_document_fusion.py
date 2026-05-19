import pytest
from langchain_core.documents import Document

from app.services.document_fusion import (
    chunk_key,
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
