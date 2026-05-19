from langchain_core.documents import Document

from app.utils.prompts import (
    _citation_marker,
    _format_page_display,
    build_prompt,
)


def test_build_prompt_includes_l_labels_and_citation_markers():
    docs = [
        Document(
            page_content="First chunk text.",
            metadata={"source": "report.pdf", "page": 2},
        ),
        Document(
            page_content="Second chunk text.",
            metadata={"source": "notes.pdf", "page": 0},
        ),
    ]

    prompt = build_prompt(docs, question="What is in the report?")

    assert 'label="L1"' in prompt
    assert 'label="L2"' in prompt
    assert "source: report.pdf" in prompt
    assert "page: 3" in prompt
    assert "【1†source=report.pdf&page=3】" in prompt
    assert "source: notes.pdf" in prompt
    assert "page: 1" in prompt
    assert "【2†source=notes.pdf&page=1】" in prompt
    assert "【N†source=<document_name>&page=<page>】" in prompt
    assert "What is in the report?" in prompt


def test_citation_marker_omits_page_when_unavailable():
    marker = _citation_marker(1, {"source": "doc.pdf"})
    assert marker == "【1†source=doc.pdf】"


def test_format_page_display_handles_string_page():
    assert _format_page_display("12") == "12"
