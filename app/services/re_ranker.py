import asyncio
import json
from app.services.llm_service import LLMService
from langchain_core.documents import Document


async def re_rank_docs(
    query: str,
    docs: list[Document],
    llm_service: LLMService,
    top_n: int = 5,
    max_candidates: int | None = None,
    max_doc_chars: int = 400,
    max_tokens: int | None = 32,
    timeout_s: float | None = 20.0,
    batch_count: int = 2,
) -> list[Document]:

    if not docs:
        return []

    candidates = docs[:max_candidates] if max_candidates is not None else docs
    if batch_count < 1:
        batch_count = 1

    batch_size = max(1, (len(candidates) + batch_count - 1) // batch_count)
    batches: list[list[Document]] = [
        candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)
    ]

    async def rerank_batch(batch_docs: list[Document]) -> list[Document]:
        blocks = []
        for i, doc in enumerate(batch_docs, start=1):
            content = doc.page_content
            if max_doc_chars:
                content = content[:max_doc_chars]
            blocks.append(
                f"""<document id=\"{i}\">
{content}
</document>"""
            )

        prompt = f"""
You are a strict ranking assistant. Select the top {top_n} documents for the question.

Question:
{query}

Documents:
{chr(10).join(blocks)}

Return only a JSON array of the best document ids in order, e.g. [2, 5, 1].
"""

        call = llm_service.generate_text_async(
            prompt,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        response = await asyncio.wait_for(call, timeout=timeout_s)
        raw = str(response.content).strip()  # type: ignore
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Invalid rerank response: {raw}")
        ids = json.loads(raw[start : end + 1])

        ranked: list[Document] = []
        for idx in ids:
            if 1 <= idx <= len(batch_docs):
                ranked.append(batch_docs[idx - 1])
            if len(ranked) >= top_n:
                break
        return ranked

    ranked_all: list[Document] = []
    for batch in batches[:batch_count]:
        ranked_all.extend(await rerank_batch(batch))

    return ranked_all[:top_n]
