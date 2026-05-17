from typing import Sequence, TypedDict, cast

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

v1 = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a question-answering assistant for a document search system.

Answer the user's question using ONLY the <context> below.
Each <document> has <metadata> (source, page) and <content>.

Rules:
- Ground every claim in the provided documents. Do not use outside knowledge.
- Do not require the question's exact phrases to appear in the text; use the relevant facts given.
- Say "I don't know." only when the context has no useful information about what was asked.
- Be concise. Cite source (and page if available) when stating facts.

<context>
{context}
</context>

Question:
{question}
"""
)


class DocumentMetadata(TypedDict, total=False):
    source: str
    page: int | str
    author: str


def build_prompt(
    docs: Sequence[Document],
    question: str,
    *,
    search_query: str | None = None,
) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        metadata = cast(DocumentMetadata, doc.metadata)  # type: ignore
        blocks.append(
            f"""<document id="{i}">
<metadata>
source: {metadata.get("source", "unknown")}
page: {metadata.get("page", "n/a")}
author: {metadata.get("author", "n/a")}
</metadata>
<content>
{doc.page_content}
</content>
</document>"""
        )
    context = "\n\n".join(blocks) if blocks else "(no documents retrieved)"

    question_block = question.strip()
    if search_query and search_query.strip() != question_block:
        question_block = (
            f"{question_block}\n"
            f"(Retrieval used a rewritten query: {search_query.strip()})"
        )

    return v1.format(context=context, question=question_block)
