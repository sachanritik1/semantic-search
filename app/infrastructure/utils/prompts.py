from typing import Sequence, TypedDict, cast

from langchain_core.documents import Document

ASK_SYSTEM_PREFIX = """You are a question-answering assistant for a document search system.

Answer the user's question using ONLY the <context> below.
Each <document> has <metadata> (source, page) and <content>.

Rules:
- Ground every claim in the provided documents. Do not use outside knowledge.
- Do not require the question's exact phrases to appear in the text; use the relevant facts given.
- Say "I don't know." only when the context has no useful information about what was asked.
- Be concise. After each factual claim, add an inline citation using the exact marker shown in that document's <citation> tag.
- Citation format: 【N†source=<document_name>&page=<page>】 where N matches the L-label number (L1 → 1, L2 → 2, …).
- Always include the document name (source) and page number from the document metadata in every citation.
- If a document has no page, use 【N†source=<document_name>】 only.
- Reuse the same citation marker when citing the same passage again."""


class DocumentMetadata(TypedDict, total=False):
    source: str
    page: int | str
    author: str


def _doc_label(index: int) -> str:
    return f"L{index}"


def _format_source(metadata: DocumentMetadata) -> str:
    source = metadata.get("source")
    if source is None:
        return "unknown"
    text = str(source).strip()
    return text or "unknown"


def _format_page_display(page: int | str | None) -> str:
    if page is None:
        return "n/a"
    if isinstance(page, int):
        return str(page + 1)
    text = str(page).strip()
    if not text or text.lower() == "n/a":
        return "n/a"
    return text


def _citation_page(page: int | str | None) -> str | None:
    if page is None:
        return None
    if isinstance(page, int):
        return str(page + 1)
    text = str(page).strip()
    if not text or text.lower() == "n/a":
        return None
    return text


def _citation_marker(index: int, metadata: DocumentMetadata) -> str:
    source = _format_source(metadata)
    page = _citation_page(metadata.get("page"))
    if page:
        return f"【{index}†source={source}&page={page}】"
    return f"【{index}†source={source}】"


def _format_document_block(index: int, doc: Document) -> str:
    metadata = cast(DocumentMetadata, doc.metadata or {})
    label = _doc_label(index)
    source = _format_source(metadata)
    page_display = _format_page_display(metadata.get("page"))
    citation = _citation_marker(index, metadata)
    return f"""<document label="{label}" id="{index}">
<metadata>
source: {source}
page: {page_display}
author: {metadata.get("author", "n/a")}
</metadata>
<citation>{citation}</citation>
<content>
{doc.page_content}
</content>
</document>"""


def _format_question_block(question: str, *, search_query: str | None = None) -> str:
    question_block = question.strip()
    if search_query and search_query.strip() != question_block:
        question_block = (
            f"{question_block}\n"
            f"(Retrieval used a rewritten query: {search_query.strip()})"
        )
    return question_block


def build_ask_messages(
    docs: Sequence[Document],
    question: str,
    *,
    search_query: str | None = None,
) -> tuple[str, str]:
    """Return (system_prefix, user_message) for provider-native prompt caching."""
    blocks = [_format_document_block(i, doc) for i, doc in enumerate(docs, start=1)]
    context = "\n\n".join(blocks) if blocks else "(no documents retrieved)"
    question_block = _format_question_block(question, search_query=search_query)
    user_message = f"""<context>
{context}
</context>

Question:
{question_block}"""
    return ASK_SYSTEM_PREFIX, user_message


def build_prompt(
    docs: Sequence[Document],
    question: str,
    *,
    search_query: str | None = None,
) -> str:
    system_prefix, user_message = build_ask_messages(
        docs,
        question,
        search_query=search_query,
    )
    return f"{system_prefix}\n\n{user_message}"
