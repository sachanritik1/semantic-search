import asyncio
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.services.llm_service import LLMService
from app.utils.llm_content import extract_rerank_payload, normalize_llm_content


@dataclass
class RerankResult:
    docs: list[Document]
    failed: bool


def _coerce_doc_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
        match = re.search(r"\d+", text)
        if match:
            return int(match.group())
    return None


def _coerce_score(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _parse_ranked_entries(raw: Any, num_docs: int) -> list[tuple[int, float]]:
    """Parse LLM output into (1-based doc id, relevance score) pairs, best first."""
    if num_docs <= 0:
        return []

    payload = extract_rerank_payload(raw)
    if not payload:
        raise ValueError(f"Invalid rerank response: {raw!r}")

    if all(isinstance(item, int) for item in payload) or all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in payload
    ):
        entries: list[tuple[int, float]] = []
        for position, item in enumerate(payload):
            doc_id = _coerce_doc_id(item)
            if doc_id is None:
                continue
            entries.append((doc_id, float(max(num_docs - position, 1))))
        return entries

    if all(isinstance(item, dict) for item in payload):
        scored: list[tuple[float, int, float]] = []
        for item in payload:
            doc_id = _coerce_doc_id(
                item.get("id")
                or item.get("document_id")
                or item.get("doc_id")
                or item.get("index")
            )
            if doc_id is None:
                continue
            score = _coerce_score(
                item.get("relevance")
                or item.get("score")
                or item.get("rating")
            )
            scored.append((score if score is not None else 0.0, doc_id, score or 0.0))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored:
            raise ValueError(f"Invalid rerank response: {raw!r}")
        return [(doc_id, relevance) for _, doc_id, relevance in scored]

    entries = []
    for position, item in enumerate(payload):
        doc_id = _coerce_doc_id(item)
        if doc_id is not None:
            entries.append((doc_id, float(max(num_docs - position, 1))))
    if not entries:
        raise ValueError(f"Invalid rerank response: {raw}")
    return entries


def _entries_degenerate(entries: list[tuple[int, float]]) -> bool:
    if not entries:
        return True
    scores = [score for _, score in entries]
    return max(scores) <= 0 or len(set(scores)) == 1


def _entries_from_fusion(batch_docs: list[Document]) -> list[tuple[int, float]]:
    """Programmatic ranking using fusion_score when LLM scores are unusable."""
    scored: list[tuple[float, int, float]] = []
    for i, doc in enumerate(batch_docs, start=1):
        meta = doc.metadata or {}
        fusion = float(meta.get("fusion_score", 0.0))
        rerank_equiv = round(fusion * 10, 2)
        scored.append((fusion, i, rerank_equiv))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [(doc_id, relevance) for _, doc_id, relevance in scored]


def _select_relevant_entries(
    entries: list[tuple[int, float]],
    top_n: int,
    min_relevance: float,
) -> list[tuple[int, float]]:
    qualified = [(doc_id, score) for doc_id, score in entries if score >= min_relevance]
    qualified.sort(key=lambda pair: pair[1], reverse=True)

    if qualified:
        return qualified[:top_n]

    fallback = sorted(entries, key=lambda pair: pair[1], reverse=True)
    return fallback[:top_n]


def _apply_ranking(
    batch_docs: list[Document],
    entries: list[tuple[int, float]],
    top_n: int,
) -> list[Document]:
    seen: set[int] = set()
    ranked: list[Document] = []

    for doc_id, score in entries:
        if doc_id in seen or not (1 <= doc_id <= len(batch_docs)):
            continue
        seen.add(doc_id)
        source = batch_docs[doc_id - 1]
        metadata = dict(source.metadata or {})
        metadata["rerank_score"] = score
        ranked.append(
            Document(page_content=source.page_content, metadata=metadata)
        )
        if len(ranked) >= top_n:
            return ranked

    return ranked


def _build_document_block(doc: Document, doc_id: int, max_doc_chars: int) -> str:
    content = doc.page_content
    if max_doc_chars:
        content = content[:max_doc_chars]

    meta = doc.metadata or {}
    fusion = meta.get("fusion_score")
    dense = meta.get("dense_norm")
    sparse = meta.get("sparse_norm")
    hints: list[str] = []
    if fusion is not None:
        hints.append(f"retrieval_score={float(fusion):.3f}")
    if dense is not None:
        hints.append(f"dense={float(dense):.3f}")
    if sparse is not None:
        hints.append(f"sparse={float(sparse):.3f}")
    hint_str = " ".join(hints)

    return f"""<document id="{doc_id}" {hint_str}>
{content}
</document>"""


def _build_rerank_prompt(query: str, blocks: list[str], num_docs: int) -> str:
    return f"""You rank pre-retrieved document chunks for a RAG system.

These {num_docs} documents were already selected by hybrid search (dense + sparse) as the best matches for the question. Rank them **relative to each other**.

<question>
{query}
</question>

<documents>
Documents:
{chr(10).join(blocks)}
</documents>

Instructions:
1. Read each document and compare it to the question.
2. Assign relevance 1-10 per id (1=weak match, 10=best match). Use the full range.
3. Higher retrieval_score usually means a stronger initial match — weigh it but verify against content.
4. The top documents should typically score 5-10; only use 0-2 for clearly unrelated chunks.
5. Return exactly {num_docs} objects, ids 1 through {num_docs}.

Output ONLY a JSON array and nothing else. Example:
[{{"id":1,"relevance":6}},{{"id":2,"relevance":9}}]
"""


async def re_rank_docs(
    query: str,
    docs: list[Document],
    llm_service: LLMService,
    top_n: int = 5,
    max_candidates: int | None = None,
    max_doc_chars: int = 400,
    max_tokens: int | None = 512,
    timeout_s: float | None = 20.0,
    batch_count: int = 1,
) -> RerankResult:
    if not docs:
        return RerankResult(docs=[], failed=False)

    candidates = docs[:max_candidates] if max_candidates is not None else docs
    if batch_count < 1:
        batch_count = 1

    batch_size = max(1, (len(candidates) + batch_count - 1) // batch_count)
    batches: list[list[Document]] = [
        candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)
    ]

    async def rerank_batch(batch_docs: list[Document]) -> list[Document]:
        blocks = [
            _build_document_block(doc, i, max_doc_chars)
            for i, doc in enumerate(batch_docs, start=1)
        ]

        prompt = _build_rerank_prompt(query, blocks, num_docs=len(batch_docs))

        call = llm_service.generate_text_async(
            prompt,
            temperature=0.1,
            max_tokens=max_tokens,
        )
        response = await asyncio.wait_for(call, timeout=timeout_s)
        raw_content = response.content  # type: ignore

        try:
            entries = _parse_ranked_entries(raw_content, len(batch_docs))
            if _entries_degenerate(entries):
                print(
                    "Rerank scores degenerate (all zero/flat); "
                    "using fusion_score order."
                )
                entries = _entries_from_fusion(batch_docs)
            selected = _select_relevant_entries(
                entries,
                top_n=top_n,
                min_relevance=settings.RERANK_MIN_RELEVANCE,
            )
        except ValueError as exc:
            preview = normalize_llm_content(raw_content)
            print(f"Rerank parse failed: {exc}; raw={preview[:500]!r}")
            entries = _entries_from_fusion(batch_docs)
            selected = _select_relevant_entries(
                entries,
                top_n=top_n,
                min_relevance=0.0,
            )

        return _apply_ranking(batch_docs, selected, top_n)

    try:
        ranked_all: list[Document] = []
        for batch in batches[:batch_count]:
            ranked_all.extend(await rerank_batch(batch))

        selected = ranked_all[:top_n]
        if not selected:
            return RerankResult(docs=[], failed=True)
        return RerankResult(docs=selected, failed=False)
    except (TimeoutError, ValueError, asyncio.TimeoutError) as exc:
        print(f"Rerank call failed: {exc}")
        return RerankResult(docs=[], failed=True)
