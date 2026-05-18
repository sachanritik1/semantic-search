import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    docs: list[Document]
    failed: bool


_model: CrossEncoder | None = None


def _tracing_enabled() -> bool:
    return bool(settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY)


def _docs_to_trace_list(docs: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": (doc.metadata or {}).get("chunk_id"),
            "document_id": (doc.metadata or {}).get("document_id"),
            "chunk_index": (doc.metadata or {}).get("chunk_index"),
            "source": (doc.metadata or {}).get("source"),
            "fusion_score": (doc.metadata or {}).get("fusion_score"),
            "content": doc.page_content,
        }
        for doc in docs
    ]


def _trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    docs = inputs.get("docs") or []
    return {
        "query": inputs.get("query"),
        "top_n": inputs.get("top_n"),
        "candidate_count": len(docs),
        "min_relevance": settings.RERANK_MIN_RELEVANCE,
        "model": settings.RERANK_MODEL_NAME,
        "input_docs": _docs_to_trace_list(docs),
    }


def _chunk_label(doc: Document, doc_id: int) -> str:
    chunk_id = (doc.metadata or {}).get("chunk_id")
    return str(chunk_id) if chunk_id is not None else f"doc-{doc_id}"


def _build_rerank_trace(
    *,
    query: str,
    docs: list[Document],
    entries: list[tuple[int, float]],
    raw_logits: dict[int, float],
    selected_ids: set[int],
    ranked: list[Document],
    status: str,
    top_n: int,
    min_relevance: float,
    error: str | None = None,
) -> dict[str, Any]:
    candidates = []
    if entries:
        for doc_id, score in entries:
            source = docs[doc_id - 1]
            meta = source.metadata or {}
            candidates.append(
                {
                    "chunk_id": meta.get("chunk_id"),
                    "document_id": meta.get("document_id"),
                    "chunk_index": meta.get("chunk_index"),
                    "source": meta.get("source"),
                    "fusion_score": meta.get("fusion_score"),
                    "rerank_logit": raw_logits.get(doc_id),
                    "rerank_score": score,
                    "selected": doc_id in selected_ids,
                    "content": source.page_content,
                }
            )
    else:
        candidates = [
            {**entry, "rerank_score": None, "selected": False}
            for entry in _docs_to_trace_list(docs)
        ]

    return {
        "status": status,
        "query": query,
        "top_n": top_n,
        "min_relevance": min_relevance,
        "candidate_count": len(docs),
        "selected_count": len(ranked),
        "candidates": candidates,
        "selected_chunk_ids": [
            (doc.metadata or {}).get("chunk_id") for doc in ranked
        ],
        "error": error,
    }


def _record_rerank_trace(payload: dict[str, Any]) -> None:
    if not _tracing_enabled():
        return
    run = get_current_run_tree()
    if run is not None:
        run.add_outputs(payload)


def _get_cross_encoder() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info("Loading cross-encoder model: %s", settings.RERANK_MODEL_NAME)
        _model = CrossEncoder(settings.RERANK_MODEL_NAME)
    return _model


def _normalize_scores(raw_logits: list[float]) -> list[float]:
    """Map cross-encoder logits to 0–10 relative to this candidate batch."""
    if not raw_logits:
        return []
    lo = min(raw_logits)
    hi = max(raw_logits)
    if hi == lo:
        return [5.0] * len(raw_logits)
    return [round(10.0 * (value - lo) / (hi - lo), 2) for value in raw_logits]


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
    candidates: list[Document],
    entries: list[tuple[int, float]],
    top_n: int,
    raw_logits: dict[int, float],
) -> list[Document]:
    seen: set[int] = set()
    ranked: list[Document] = []

    for doc_id, score in entries:
        if doc_id in seen or not (1 <= doc_id <= len(candidates)):
            continue
        seen.add(doc_id)
        source = candidates[doc_id - 1]
        metadata = dict(source.metadata or {})
        metadata["rerank_logit"] = raw_logits.get(doc_id)
        metadata["rerank_score"] = score
        ranked.append(
            Document(page_content=source.page_content, metadata=metadata)
        )
        if len(ranked) >= top_n:
            break

    return ranked


def _score_candidates(
    query: str,
    candidates: list[Document],
) -> tuple[list[tuple[int, float]], dict[int, float]]:
    pairs = [(query, doc.page_content) for doc in candidates]
    logits = _get_cross_encoder().predict(pairs)
    raw_list = [float(value) for value in logits]
    normalized = _normalize_scores(raw_list)
    raw_by_id = {doc_id: raw for doc_id, raw in enumerate(raw_list, start=1)}
    entries = sorted(
        enumerate(normalized, start=1),
        key=lambda item: raw_by_id[item[0]],
        reverse=True,
    )
    return [(doc_id, score) for doc_id, score in entries], raw_by_id


@traceable(
    run_type="chain",
    name="cross_encoder_rerank",
    process_inputs=_trace_inputs,
)
def re_rank_docs(
    query: str,
    docs: list[Document],
    top_n: int = 5,
) -> RerankResult:
    if not docs:
        logger.warning("Rerank skipped: no candidates")
        _record_rerank_trace(
            _build_rerank_trace(
                query=query,
                docs=[],
                entries=[],
                raw_logits={},
                selected_ids=set(),
                ranked=[],
                status="skipped",
                top_n=top_n,
                min_relevance=settings.RERANK_MIN_RELEVANCE,
            )
        )
        return RerankResult(docs=[], failed=False)

    min_relevance = settings.RERANK_MIN_RELEVANCE
    logger.info(
        "Reranking %d candidates (top_n=%d, min_relevance=%.1f)",
        len(docs),
        top_n,
        min_relevance,
    )

    entries: list[tuple[int, float]] = []
    raw_logits: dict[int, float] = {}
    selected_ids: set[int] = set()
    try:
        entries, raw_logits = _score_candidates(query, docs)
        raw_values = list(raw_logits.values())
        logger.info(
            "Rerank raw logits: min=%.3f max=%.3f mean=%.3f",
            min(raw_values),
            max(raw_values),
            sum(raw_values) / len(raw_values),
        )
        selected = _select_relevant_entries(
            entries,
            top_n=top_n,
            min_relevance=min_relevance,
        )
        selected_ids = {doc_id for doc_id, _ in selected}

        used_fallback = not any(score >= min_relevance for _, score in entries)
        if used_fallback:
            logger.warning(
                "Rerank fallback: no scores >= %.1f, taking top %d",
                min_relevance,
                len(selected),
            )

        ranked = _apply_ranking(docs, selected, top_n, raw_logits)
        if not ranked:
            logger.warning("Rerank failed: no documents after ranking")
            _record_rerank_trace(
                _build_rerank_trace(
                    query=query,
                    docs=docs,
                    entries=entries,
                    raw_logits=raw_logits,
                    selected_ids=selected_ids,
                    ranked=[],
                    status="failed",
                    top_n=top_n,
                    min_relevance=min_relevance,
                    error="no_documents_after_ranking",
                )
            )
            return RerankResult(docs=[], failed=True)

        logger.info("Rerank ok: %d/%d candidates selected", len(ranked), len(docs))
        _record_rerank_trace(
            _build_rerank_trace(
                query=query,
                docs=docs,
                entries=entries,
                raw_logits=raw_logits,
                selected_ids=selected_ids,
                ranked=ranked,
                status="fallback" if used_fallback else "success",
                top_n=top_n,
                min_relevance=min_relevance,
            )
        )
        return RerankResult(docs=ranked, failed=False)
    except Exception as exc:
        logger.exception("Rerank failed: cross-encoder error")
        _record_rerank_trace(
            _build_rerank_trace(
                query=query,
                docs=docs,
                entries=entries,
                raw_logits=raw_logits,
                selected_ids=selected_ids,
                ranked=[],
                status="error",
                top_n=top_n,
                min_relevance=min_relevance,
                error=str(exc),
            )
        )
        return RerankResult(docs=[], failed=True)
