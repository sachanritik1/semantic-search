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
            if doc_id is not None:
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
    scored = [
        (float((doc.metadata or {}).get("fusion_score", 0.0)), i)
        for i, doc in enumerate(batch_docs, start=1)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [(i, round(fusion * 10, 2)) for fusion, i in scored]


def _select_relevant_entries(
    entries: list[tuple[int, float]],
    top_n: int,
    min_relevance: float,
) -> list[tuple[int, float]]:
    qualified = sorted(
        [(doc_id, score) for doc_id, score in entries if score >= min_relevance],
        key=lambda pair: pair[1],
        reverse=True,
    )
    if qualified:
        return qualified[:top_n]
    return sorted(entries, key=lambda pair: pair[1], reverse=True)[:top_n]


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
            break

    return ranked


def _doc_block(doc: Document, doc_id: int, max_doc_chars: int) -> str:
    content = doc.page_content[:max_doc_chars] if max_doc_chars else doc.page_content
    fusion = (doc.metadata or {}).get("fusion_score")
    score_attr = f' fusion="{float(fusion):.2f}"' if fusion is not None else ""
    return f'<doc id="{doc_id}"{score_attr}>\n{content}\n</doc>'


def _rerank_prompt(query: str, blocks: list[str], num_docs: int) -> str:
    return (
        f"Score each doc 1-{num_docs} for relevance to the question (1-10, relative ranking).\n"
        f"Question: {query}\n\n"
        f"{chr(10).join(blocks)}\n\n"
        f'Return JSON only, one entry per id: [{{"id":1,"relevance":N}}, ...]'
    )


def _resolve_entries(
    raw_content: Any,
    batch_docs: list[Document],
    top_n: int,
) -> list[tuple[int, float]]:
    try:
        entries = _parse_ranked_entries(raw_content, len(batch_docs))
        if _entries_degenerate(entries):
            print(
                "Rerank scores degenerate (all zero/flat); using fusion_score order."
            )
            entries = _entries_from_fusion(batch_docs)
        return _select_relevant_entries(
            entries,
            top_n=top_n,
            min_relevance=settings.RERANK_MIN_RELEVANCE,
        )
    except ValueError as exc:
        preview = normalize_llm_content(raw_content)
        print(f"Rerank parse failed: {exc}; raw={preview[:500]!r}")
        return _select_relevant_entries(
            _entries_from_fusion(batch_docs),
            top_n=top_n,
            min_relevance=0.0,
        )


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
    batch_count = max(1, batch_count)
    batch_size = max(1, (len(candidates) + batch_count - 1) // batch_count)
    batches = [
        candidates[i : i + batch_size]
        for i in range(0, len(candidates), batch_size)
    ][:batch_count]

    async def rerank_batch(batch_docs: list[Document]) -> list[Document]:
        blocks = [
            _doc_block(doc, i, max_doc_chars)
            for i, doc in enumerate(batch_docs, start=1)
        ]
        response = await asyncio.wait_for(
            llm_service.generate_text_async(
                _rerank_prompt(query, blocks, len(batch_docs)),
                temperature=0.1,
                max_tokens=max_tokens,
            ),
            timeout=timeout_s,
        )
        selected = _resolve_entries(response.content, batch_docs, top_n)
        return _apply_ranking(batch_docs, selected, top_n)

    try:
        ranked: list[Document] = []
        for batch in batches:
            ranked.extend(await rerank_batch(batch))
        selected = ranked[:top_n]
        if not selected:
            print("Rerank returned no documents.")
        return RerankResult(docs=selected, failed=not selected)
    except (TimeoutError, asyncio.TimeoutError) as exc:
        print(f"Rerank call failed: {exc}")
        return RerankResult(docs=[], failed=True)
