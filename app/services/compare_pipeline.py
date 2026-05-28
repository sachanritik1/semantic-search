import asyncio
import json

from app.db.weaviate_store import any_chunks_exist, bm25_search, dense_search
from app.services.embedder import get_embeddings
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
            f"""<document id="dense-{item["index"]}">
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
            f"""<document id="sparse-{item["index"]}">
<source>sparse</source>
<bm25_score>{item.get("score")}</bm25_score>
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


class Retriever:
    def retrieve(self, question: str, top_k: int = 5) -> dict:
        embeddings = get_embeddings()
        query_embedding = embeddings.embed_query(question)

        dense_results_raw = dense_search(query_embedding, limit=top_k)
        dense_results = [
            {
                "index": i,
                "content": doc.page_content,
                "metadata": getattr(doc, "metadata", None) or {},
            }
            for i, doc in enumerate([d for d, _ in dense_results_raw])
        ]

        if not any_chunks_exist():
            return {"dense": dense_results, "sparse": []}

        sparse_results_raw = bm25_search(question, limit=top_k)

        sparse_results = [
            {
                "index": i,
                "score": score,
                "content": doc.page_content,
                "document_id": doc.metadata.get("document_id", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "source": doc.metadata.get("source", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "metadata": doc.metadata or {},
            }
            for i, (doc, score) in enumerate(sparse_results_raw)
        ]

        return {"dense": dense_results, "sparse": sparse_results}


class LLMScorer:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def score(
        self,
        question: str,
        retrieval: dict,
        *,
        timeout_s: float = 45.0,
        max_tokens: int = 1024,
    ) -> dict:
        dense = retrieval["dense"]
        sparse = retrieval["sparse"]

        if not dense and not sparse:
            return {
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

        return {"llm_comparison": llm_comparison}


class ComparePipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm_scorer: LLMScorer | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.llm_scorer = llm_scorer

    async def compare(
        self,
        question: str,
        top_k: int = 5,
        *,
        with_llm: bool = False,
    ) -> dict:
        results = self.retriever.retrieve(question, top_k=top_k)
        if with_llm and self.llm_scorer is not None:
            llm_result = await self.llm_scorer.score(question, results)
            results = {**results, **llm_result}
        return results
