from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
import asyncio


async def score_doc(query: str, doc: Document, llm: BaseChatModel)-> tuple[int, Document]:
    prompt = f"""
        Score the relevance of the following document to the question on a scale of 1 to 5.

        Question:
        {query}

        Document:
        {doc.page_content}

        Score only with a number.
        """
    response = await llm.ainvoke(prompt)
    score = int(str(response.content).strip()) # type: ignore
    return score, doc



async def re_rank_docs(
    query: str,
    docs: list[Document],
    llm: BaseChatModel,
    top_n: int = 5
) -> list[Document]:

    tasks = [
        score_doc(query, doc, llm)
        for doc in docs
    ]

    scored = await asyncio.gather(*tasks)

    scored.sort(reverse=True, key=lambda x: x[0])

    res = [doc for _, doc in scored[:top_n]]
    print(f"Scores of top {top_n} documents:")
    for i, (score, doc) in enumerate(scored[:top_n]):
        print(f"Document {i+1}: Score {score} Content: {doc.page_content}")
    return res