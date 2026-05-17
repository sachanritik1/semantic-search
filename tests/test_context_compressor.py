from langchain_core.documents import Document

from app.services.context_compressor import compress_documents_for_context, max_chars_for_doc


def test_max_chars_high_tier():
    doc = Document(page_content="x", metadata={"rerank_score": 9})
    assert max_chars_for_doc(doc, rank=3) >= 800


def test_max_chars_low_tier():
    doc = Document(page_content="x", metadata={"rerank_score": 2})
    assert max_chars_for_doc(doc, rank=5) == 150


def test_compress_truncates_long_content():
    doc = Document(
        page_content="a" * 1000,
        metadata={"rerank_score": 9},
    )
    compressed = compress_documents_for_context([doc])
    assert compressed[0].page_content.endswith("…")
    assert len(compressed[0].page_content) < 1000


def test_compress_leaves_short_content():
    doc = Document(page_content="short", metadata={"rerank_score": 2})
    compressed = compress_documents_for_context([doc])
    assert compressed[0].page_content == "short"
