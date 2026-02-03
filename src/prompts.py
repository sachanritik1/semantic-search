from typing import Sequence, TypedDict, cast

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

v1 = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a question-answering assistant. \

    Answer the question using ONLY the information provided inside the <context> block. \
    Each document in the context is enclosed in <document> tags and may contain <metadata> and <content> sections. \

    Rules:
    - If the answer is not explicitly present in the context, respond with: "I don't know." \
    - Do NOT use any external knowledge. \
    - Do NOT make assumptions or guesses. \
    - Do NOT add explanations beyond what is asked. \
    - When stating a fact, cite the source using the source or page from metadata. \

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

def build_prompt(docs: Sequence[Document], question: str) -> str:
    
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        metadata = cast(DocumentMetadata, doc.metadata) # type: ignore
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
    context  = "\n\n".join(blocks)

    return v1.format(
        context=context,
        question=question
    )
