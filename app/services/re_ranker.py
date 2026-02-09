import asyncio
from app.services.llm_service import LLMService
from langchain_core.documents import Document


async def score_doc(query: str, doc: Document, llm_service: LLMService)-> tuple[int, Document]:
    prompt = f"""
        Score the relevance of the following document to the question on a scale of 1 to 5.

        Question:
        {query}

        Document:
        {doc.page_content}

        Score only with a number.
        """
    response = await llm_service.generate_text_async(prompt)
    score = int(str(response.content).strip()) # type: ignore
    return score, doc


async def re_rank_docs(
    query: str,
    docs: list[Document],
    llm_service: LLMService,
    top_n: int = 5,
) -> list[Document]:

    tasks = [
        score_doc(query, doc, llm_service)
        for doc in docs
    ]

    scored = await asyncio.gather(*tasks)

    scored.sort(reverse=True, key=lambda x: x[0])

    res = [doc for _, doc in scored[:top_n]]

    print(f"Scores of top {top_n} documents:")
    for i, (score, _) in enumerate(scored[:top_n]):
        print(f"Document {i+1}: Score {score}")

    return res
