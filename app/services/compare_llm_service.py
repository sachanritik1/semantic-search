import asyncio
import json

from app.services.compare_service import CompareService
from app.services.llm_service import LLMService
from app.utils.prompt_cache import cache_key

MAX_DOC_CHARS = 500

_COMPARE_SYSTEM_PROMPT = """You are a retrieval evaluation assistant. Score how relevant each document is to answering the question.

For every document id listed above, rate relevance from 0 (irrelevant) to 10 (highly relevant) and give a one-sentence reason.
Then compare the two retrievers (dense vector vs sparse BM25): which found better matches overall and why.

Return only valid JSON in this shape:
{
  "document_scores": [
    {"id": "dense-0", "source": "dense", "relevance": 8, "reason": "..."}
  ],
  "retriever_verdict": {
    "winner": "dense" | "sparse" | "tie",
    "dense_strength": "...",
    "sparse_strength": "..."
  },
  "summary": "..."
}"""


def _format_compare_documents(dense: list[dict], sparse: list[dict]) -> str:
    blocks: list[str] = []

    for item in dense:
        content = item["content"]
        if len(content) > MAX_DOC_CHARS:
            content = content[:MAX_DOC_CHARS] + "..."
        blocks.append(
            f"""<document id="dense-{item['index']}">
<source>dense</source>
<content>
{content}
</content>
</document>"""
        )

    for item in sparse:
        content = item["content"]
        if len(content) > MAX_DOC_CHARS:
            content = content[:MAX_DOC_CHARS] + "..."
        blocks.append(
            f"""<document id="sparse-{item['index']}">
<source>sparse</source>
<bm25_score>{item.get('score')}</bm25_score>
<content>
{content}
</content>
</document>"""
        )

    return "\n".join(blocks)


def _build_compare_messages(
    question: str,
    dense: list[dict],
    sparse: list[dict],
) -> tuple[str, str]:
    documents = _format_compare_documents(dense, sparse)
    user_message = f"""Question:
{question}

Documents:
{documents}"""
    return _COMPARE_SYSTEM_PROMPT, user_message


def _build_compare_prompt(
    question: str,
    dense: list[dict],
    sparse: list[dict],
) -> str:
    system_prompt, user_message = _build_compare_messages(question, dense, sparse)
    return f"{system_prompt}\n\n{user_message}"


def _parse_llm_comparison(raw: str) -> dict:
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Invalid LLM comparison response: {raw}")
    return json.loads(text[start : end + 1])


class CompareLLMService:
    def __init__(self, compare_service: CompareService, llm_service: LLMService):
        self.compare_service = compare_service
        self.llm_service = llm_service

    async def compare_with_llm(
        self,
        question: str,
        top_k: int = 5,
        *,
        timeout_s: float = 45.0,
        max_tokens: int = 1024,
    ) -> dict:
        retrieval = self.compare_service.compare(question, top_k=top_k)
        dense = retrieval["dense"]
        sparse = retrieval["sparse"]

        if not dense and not sparse:
            return {
                **retrieval,
                "llm_comparison": {
                    "document_scores": [],
                    "retriever_verdict": {
                        "winner": "tie",
                        "dense_strength": "No documents retrieved.",
                        "sparse_strength": "No documents retrieved.",
                    },
                    "summary": "No documents available to compare.",
                },
            }

        system_prompt, user_message = _build_compare_messages(question, dense, sparse)
        call = self.llm_service.generate_text_async(
            user_message,
            temperature=0.0,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            cache_key=cache_key("compare"),
        )
        response = await asyncio.wait_for(call, timeout=timeout_s)
        llm_comparison = _parse_llm_comparison(str(response.content))

        return {**retrieval, "llm_comparison": llm_comparison}
