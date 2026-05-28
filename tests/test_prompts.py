from langchain_core.documents import Document

from app.infrastructure.utils.prompts import (
    ASK_SYSTEM_PREFIX,
    _citation_marker,
    _format_page_display,
    build_ask_messages,
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


def test_build_ask_messages_stable_prefix():
    docs_a = [Document(page_content="alpha", metadata={"source": "a.pdf"})]
    docs_b = [Document(page_content="beta", metadata={"source": "b.pdf"})]

    prefix_a, user_a = build_ask_messages(docs_a, question="What is alpha?")
    prefix_b, user_b = build_ask_messages(docs_b, question="What is beta?")

    assert prefix_a == prefix_b == ASK_SYSTEM_PREFIX
    assert "alpha" in user_a
    assert "beta" in user_b
    assert "What is alpha?" in user_a
    assert "What is beta?" in user_b
    assert "<context>" in user_a
    assert ASK_SYSTEM_PREFIX not in user_a


def test_build_ask_messages_user_message_contains_context_and_question():
    docs = [Document(page_content="chunk", metadata={"source": "doc.pdf"})]
    _, user_message = build_ask_messages(
        docs,
        question="Summarize?",
        search_query="summary overview",
    )

    assert "<context>" in user_message
    assert "chunk" in user_message
    assert "Summarize?" in user_message
    assert "summary overview" in user_message
